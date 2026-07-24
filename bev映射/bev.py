"""Multi-camera ground-plane BEV generation for the bundled nuScenes images."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
#OpenCV 读图片，投影，保存图片
import cv2
#矩阵运算
import numpy as np


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
#时间戳正则
TIMESTAMP_RE = re.compile(r"__(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

#四元数转旋转矩阵 把相机坐标系的方向，变成车辆坐标系的方向
def quaternion_matrix(q: list[float]) -> np.ndarray:
    """Return a camera-to-ego rotation matrix for a [w, x, y, z] quaternion."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = w * w + x * x + y * y + z * z
    if n < 1e-16:
        raise ValueError("Invalid zero quaternion")
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

#保存摄像头标定
@dataclass(frozen=True)
class Calibration:
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

#加载标定 读取json文件
def load_calibrations(
    path: Path, sensor_path: Path, scene_name: str
) -> dict[str, Calibration]:
    """Load a camera rig, resolving records through sensor_token."""
    sensors = json.loads(sensor_path.read_text(encoding="utf-8"))
    channel_by_token = {
        record["token"]: record["channel"]
        for record in sensors
        if record.get("modality") == "camera"
    }
    missing_channels = set(CAMERAS) - set(channel_by_token.values())
    if missing_channels:
        raise ValueError(
            f"Missing camera channels in {sensor_path}: {', '.join(sorted(missing_channels))}"
        )

    records = json.loads(path.read_text(encoding="utf-8"))
    camera_records = [
        r
        for r in records
        if r.get("camera_intrinsic") and r.get("sensor_token") in channel_by_token
    ]
    if len(camera_records) < 6:
        raise ValueError(f"No six-camera calibration found in {path}")

    # The table contains the n008 rig first and n015 rig second. Later entries
    # repeat these physical rigs for other logs.
    block_start = 6 if scene_name.startswith("n015-") else 0
    block = camera_records[block_start : block_start + 6]
    if len(block) != 6:
        raise ValueError("Incomplete camera calibration block")
    calibrations = {
        channel_by_token[record["sensor_token"]]: Calibration(
            intrinsic=np.asarray(record["camera_intrinsic"], dtype=np.float64),
            rotation=quaternion_matrix(record["rotation"]),
            translation=np.asarray(record["translation"], dtype=np.float64),
        )
        for record in block
    }
    if set(calibrations) != set(CAMERAS):
        raise ValueError("Selected calibration block does not contain six unique cameras")
    return calibrations

#从图片名字中提取时间戳
def timestamp(path: Path) -> int:
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot read timestamp from {path.name}")
    return int(match.group(1))

#从图片名字中提取场景名
def scene_name(path: Path) -> str:
    return path.name.split("__", 1)[0]

#！返回一帧对应的六张图片
def synchronized_frames(input_dir: Path, scene: str | None, tolerance_ms: float):
    files = {
        camera: sorted((input_dir / camera).glob("*.jpg"), key=timestamp)
        for camera in CAMERAS
    }
    #检查是否缺摄像头
    if any(not values for values in files.values()):
        missing = [camera for camera, values in files.items() if not values]
        raise FileNotFoundError(f"Missing images for: {', '.join(missing)}")
    #找有哪些scene
    available = sorted({scene_name(p) for p in files["CAM_FRONT"]})
    #选择当前场景
    selected = scene or available[0]
    if selected not in available:
        raise ValueError(f"Unknown scene {selected!r}. Available: {', '.join(available)}")
    #按场景过滤图片
    per_camera = {
        camera: [p for p in values if scene_name(p) == selected]
        for camera, values in files.items()
    }
    #保存每个摄像头时间戳
    indexes = {
        camera: np.asarray([timestamp(p) for p in values], dtype=np.int64)
        for camera, values in per_camera.items()
    }
    tolerance_us = int(tolerance_ms * 1000)
    #时间匹配 
    #以前视摄像头为基准
    for front in per_camera["CAM_FRONT"]:
        target = timestamp(front)
        frame: dict[str, Path] = {}
        valid = True
        #对其他摄像头寻找最接近的时间
        for camera in CAMERAS:
            ts = indexes[camera]
            pos = int(np.searchsorted(ts, target))
            candidates = [i for i in (pos - 1, pos) if 0 <= i < len(ts)]
            best = min(candidates, key=lambda i: abs(int(ts[i]) - target))
            if abs(int(ts[best]) - target) > tolerance_us:
                valid = False
                break
            frame[camera] = per_camera[camera][best]
        if valid:
            yield selected, target, frame


class BEVProjector:
    def __init__(
        self,
        calibrations: dict[str, Calibration],
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: float,
    ):
        self.calibrations = calibrations
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.resolution = resolution
        self.width = int(round((y_max - y_min) / resolution))
        self.height = int(round((x_max - x_min) / resolution))
        xs = x_max - (np.arange(self.height, dtype=np.float32) + 0.5) * resolution
        ys = y_max - (np.arange(self.width, dtype=np.float32) + 0.5) * resolution
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        self.ground = np.stack(
        (
            xx,
            yy,
            np.zeros_like(xx)
        ),
        axis=-1
    ).reshape(-1,3)

    def _maps(self, camera: str, image_shape: tuple[int, int]):
        calibration = self.calibrations[camera]
        camera_points = (self.ground - calibration.translation) @ calibration.rotation
        
        depth = camera_points[:, 2]

        projected = camera_points @ calibration.intrinsic.T
        u = projected[:, 0] / np.maximum(projected[:, 2], 1e-8)
        v = projected[:, 1] / np.maximum(projected[:, 2], 1e-8)
        h, w = image_shape
        valid = (depth > 0.1) & (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)

        # Use a broad smooth overlap. In particular, weighting by distance from
        # the optical axis prevents the front/side camera boundary from becoming
        # a hard diagonal seam.
        edge = np.minimum.reduce((u, w - 1 - u, v, h - 1 - v))
        feather = np.clip(edge / 180.0, 0.0, 1.0)
        feather = feather * feather * (3.0 - 2.0 * feather)
        cx = calibration.intrinsic[0, 2]
        optical = np.clip(1.0 - np.abs(u - cx) / (0.58 * w), 0.0, 1.0)
        optical = optical * optical * (3.0 - 2.0 * optical)
        ray_norm = np.linalg.norm(camera_points, axis=1)
        facing = np.clip(depth / np.maximum(ray_norm, 1e-6), 0.0, 1.0)
        weight = np.where(valid, feather * optical * facing, 0.0).astype(np.float32)
        return (
            u.reshape(self.height, self.width).astype(np.float32),
            v.reshape(self.height, self.width).astype(np.float32),
            weight.reshape(self.height, self.width),
        )

    def project(self, images: dict[str, np.ndarray]) -> np.ndarray:
        accumulator = np.zeros((self.height, self.width, 3), dtype=np.float32)
        weights = np.zeros((self.height, self.width), dtype=np.float32)
        for camera in CAMERAS:
            image = images[camera]
            map_x, map_y, weight = self._maps(camera, image.shape[:2])
            warped = cv2.remap(
                image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            ).astype(np.float32)
            accumulator += warped * weight[..., None]
            weights += weight
        bev = accumulator / np.maximum(weights[..., None], 1e-6)
        bev[weights == 0] = (28, 28, 28)
        return np.clip(bev, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project six synchronized cameras to BEV.")
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--calibration", type=Path, default=Path("calibrated_sensor.json"))
    parser.add_argument("--sensors", type=Path, default=Path("sensor.json"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--scene", help="Scene prefix; defaults to the first available scene")
    parser.add_argument("--frame", type=int, default=None, help="Only process this zero-based frame")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tolerance-ms", type=float, default=60.0)
    parser.add_argument("--x-range", nargs=2, type=float, default=(-10.0, 30.0), metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", nargs=2, type=float, default=(-15.0, 15.0), metavar=("MIN", "MAX"))
    parser.add_argument("--resolution", type=float, default=0.05, help="Metres per pixel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = synchronized_frames(args.input, args.scene, args.tolerance_ms)
    written = 0
    projector = None
    for index, (scene, ts, paths) in enumerate(frames):
        if args.frame is not None and index != args.frame:
            continue
        if projector is None:
            projector = BEVProjector(
                load_calibrations(args.calibration, args.sensors, scene),
                *args.x_range,
                *args.y_range,
                args.resolution,
            )
        images = {camera: cv2.imread(str(path)) for camera, path in paths.items()}
        unreadable = [camera for camera, image in images.items() if image is None]
        if unreadable:
            raise OSError(f"Could not read: {', '.join(unreadable)}")
        bev = projector.project(images)
        target_dir = args.output / scene
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"bev_{index:04d}_{ts}.jpg"
        if not cv2.imwrite(str(target), bev, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write {target}")
        print(f"[{index:04d}] {target}")
        written += 1
        if args.frame is not None or (args.max_frames is not None and written >= args.max_frames):
            break
    if written == 0:
        raise RuntimeError("No synchronized frames matched the requested options")


if __name__ == "__main__":
    main()
