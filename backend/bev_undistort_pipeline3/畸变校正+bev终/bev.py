"""Six-camera synchronization, calibration loading, and ground-plane BEV.
同步接口；标定接口；BEV 投影器；坐标转换接口"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ("front_center", "front_right", "side_right", "rear_center", "side_left", "front_left")
FRAME_RE = re.compile(r"_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
CAMERA_DIRS = {camera: f"cam_{camera}" for camera in CAMERAS}


def quaternion_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = w*w + x*x + y*y + z*z
    if n < 1e-16:
        raise ValueError("Invalid zero quaternion")
    s = 2.0 / n
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ], dtype=np.float64)


@dataclass(frozen=True)
class Calibration:
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

#读取原始内外参
def _view_rotation(view: dict) -> np.ndarray:
    """Return OpenCV-camera to vehicle rotation from an A2D2 view.

    A2D2 view axes are forward/left/up, while pinhole projection expects
    right/down/forward.  The column conversion below bridges those frames.
    """
    forward = np.asarray(view["x-axis"], dtype=np.float64)
    left = np.asarray(view["y-axis"], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.cross(forward, left); up /= np.linalg.norm(up)
    left = np.cross(up, forward); left /= np.linalg.norm(left)
    return np.column_stack((-left, -up, forward))


def load_calibrations(path: Path) -> dict[str, Calibration]:
    records = json.loads(path.read_text(encoding="utf-8"))["cameras"]
    calibrations = {
        camera: Calibration(
            np.asarray(item["CamMatrix"], dtype=np.float64),
            _view_rotation(item["view"]),
            np.asarray(item["view"]["origin"], dtype=np.float64),
        ) for camera, item in records.items() if camera in CAMERAS
    }
    missing = set(CAMERAS) - set(calibrations)
    if missing:
        raise ValueError(f"Incomplete calibration; missing: {', '.join(sorted(missing))}")
    return calibrations

#与畸变校正连接，接收矫正后新内参，传入bev
def with_intrinsics(calibrations: dict[str, Calibration], intrinsics: dict[str, np.ndarray]):
    return {camera: replace(calib, intrinsic=np.asarray(intrinsics[camera], dtype=np.float64)) for camera, calib in calibrations.items()}


def timestamp(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot read timestamp from {path.name}")
    return int(match.group(1))

#对齐六路时间
def _sequence_root(input_dir: Path) -> Path:
    if (input_dir / "camera_lidar").is_dir():
        return input_dir / "camera_lidar"
    return input_dir


def synchronized_frames(input_dir: Path, scene: str | None = None, tolerance_ms: float = 0.0):
    """Yield A2D2 frames synchronized by the sample ID embedded in filenames."""
    root = _sequence_root(input_dir)
    available = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "camera").is_dir())
    if not available:
        raise FileNotFoundError(f"No A2D2 sequence found below {root}")
    selected = scene or available[0]
    camera_root = root / selected / "camera"
    files = {
        c: {timestamp(p): p for p in (camera_root / CAMERA_DIRS[c]).glob("*.png")}
        for c in CAMERAS
    }
    missing = [c for c, values in files.items() if not values]
    if missing:
        raise FileNotFoundError(f"Missing A2D2 images for: {', '.join(missing)}")
    common_ids = sorted(set.intersection(*(set(values) for values in files.values())))
    if not common_ids:
        raise ValueError(f"No synchronized six-camera frames in sequence {selected}")
    for frame_id in common_ids:
        yield selected, frame_id, {c: files[c][frame_id] for c in CAMERAS}


class BEVProjector:
    def __init__(self, calibrations, x_min=-10.0, x_max=30.0, y_min=-15.0, y_max=15.0, resolution=0.05):
        self.calibrations, self.x_min, self.x_max = calibrations, x_min, x_max
        self.y_min, self.y_max, self.resolution = y_min, y_max, resolution
        self.width = int(round((y_max-y_min)/resolution)); self.height = int(round((x_max-x_min)/resolution))
        xs = x_max-(np.arange(self.height, dtype=np.float32)+.5)*resolution
        ys = y_max-(np.arange(self.width, dtype=np.float32)+.5)*resolution
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        self.ground = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(-1, 3)
    #内部映射接口：读取相机标定，车辆地面点转到相机坐标系，获取深度，使用内参投影
    def _maps(self, camera, shape):
        c = self.calibrations[camera]
        points = (self.ground-c.translation) @ c.rotation
        depth = points[:, 2]; projected = points @ c.intrinsic.T
        u = projected[:, 0]/np.maximum(projected[:, 2], 1e-8); v = projected[:, 1]/np.maximum(projected[:, 2], 1e-8)
        h, w = shape
        valid = (depth > .1) & (u >= 1) & (u < w-2) & (v >= 1) & (v < h-2)
        edge = np.minimum.reduce((u, w-1-u, v, h-1-v))
        feather = np.clip(edge/180., 0., 1.); feather = feather*feather*(3.-2.*feather)
        optical = np.clip(1.-np.abs(u-c.intrinsic[0,2])/(.58*w), 0., 1.); optical = optical*optical*(3.-2.*optical)
        facing = np.clip(depth/np.maximum(np.linalg.norm(points, axis=1), 1e-6), 0., 1.)
        weight = np.where(valid, feather*optical*facing, 0.).astype(np.float32)
        shape2 = (self.height, self.width)
        return u.reshape(shape2).astype(np.float32), v.reshape(shape2).astype(np.float32), weight.reshape(shape2)
    #bev生成接口
    def project(self, images):
        acc = np.zeros((self.height, self.width, 3), np.float32); weights = np.zeros((self.height, self.width), np.float32)
        for camera in CAMERAS:
            mx, my, weight = self._maps(camera, images[camera].shape[:2])
            warped = cv2.remap(images[camera], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
            acc += warped*weight[..., None]; weights += weight
        out = acc/np.maximum(weights[..., None], 1e-6); out[weights == 0] = (28, 28, 28)
        return np.clip(out, 0, 255).astype(np.uint8)
    #坐标转换接口：车辆坐标系中的米制位置转换为BEV图片像素
    def ground_to_pixel(self, x, y):
        row = int(round((self.x_max-x)/self.resolution-.5)); col = int(round((self.y_max-y)/self.resolution-.5))
        return max(0, min(self.width-1, col)), max(0, min(self.height-1, row))
