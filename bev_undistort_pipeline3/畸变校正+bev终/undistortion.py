"""Image undistortion with an explicit image-plus-intrinsics interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class UndistortResult:
    image: np.ndarray
    intrinsic: np.ndarray


@dataclass(frozen=True)
class CameraDistortion:
    model: str
    intrinsic: np.ndarray
    coefficients: np.ndarray
    balance: float = 0.0
    output_intrinsic: np.ndarray | None = None

#读取畸变参数
def load_distortions(path: Path) -> dict[str, CameraDistortion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # A2D2 keeps intrinsics, lens model and distortion in cams_lidars.json.
    if "cameras" in raw:
        result = {}
        for camera, item in raw["cameras"].items():
            lens = str(item.get("Lens", "")).lower()
            model = "fisheye" if lens == "fisheye" else "pinhole"
            result[camera] = CameraDistortion(
                model=model,
                intrinsic=np.asarray(item["CamMatrixOriginal"], dtype=np.float64),
                coefficients=np.asarray(item["Distortion"], dtype=np.float64).reshape(-1, 1),
                output_intrinsic=np.asarray(item["CamMatrix"], dtype=np.float64),
            )
        return result
    result: dict[str, CameraDistortion] = {}
    for camera, item in raw.items():
        model = str(item.get("model", "pinhole")).lower()
        if model not in {"pinhole", "fisheye"}:
            raise ValueError(f"{camera}: model must be pinhole or fisheye")
        k = np.asarray(item["intrinsic"], dtype=np.float64)
        d = np.asarray(item["distortion"], dtype=np.float64).reshape(-1, 1)
        if k.shape != (3, 3):
            raise ValueError(f"{camera}: intrinsic must be 3x3")
        if model == "fisheye" and d.size != 4:
            raise ValueError(f"{camera}: fisheye distortion must contain 4 values")
        result[camera] = CameraDistortion(model, k, d, float(item.get("balance", 0.0)))
    return result


class Undistorter:
    """Caches remap tables and returns the matching output intrinsic matrix."""

    def __init__(self, configurations: dict[str, CameraDistortion]):
        self.configurations = configurations
        self._cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _maps(self, camera: str, size: tuple[int, int]):
        key = (camera, *size)
        if key in self._cache:
            return self._cache[key]
        if camera not in self.configurations:
            raise KeyError(f"No distortion calibration for {camera}")
        cfg = self.configurations[camera]
        w, h = size
        if cfg.model == "fisheye":
            new_k = cfg.output_intrinsic
            if new_k is None:
                new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    cfg.intrinsic, cfg.coefficients, size, np.eye(3), balance=cfg.balance
                )
            map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                cfg.intrinsic, cfg.coefficients, np.eye(3), new_k, size, cv2.CV_32FC1
            )
        else:
            new_k = cfg.output_intrinsic
            if new_k is None:
                new_k, _ = cv2.getOptimalNewCameraMatrix(
                    cfg.intrinsic, cfg.coefficients, size, cfg.balance, size
                )
            map_x, map_y = cv2.initUndistortRectifyMap(
                cfg.intrinsic, cfg.coefficients, None, new_k, size, cv2.CV_32FC1
            )
        value = (map_x, map_y, new_k)
        self._cache[key] = value
        return value

    def process(self, camera: str, image: np.ndarray) -> UndistortResult:
        if image is None or image.ndim < 2:
            raise ValueError(f"{camera}: invalid image")
        h, w = image.shape[:2]
        map_x, map_y, new_k = self._maps(camera, (w, h))
        corrected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return UndistortResult(corrected, new_k.copy())

    def process_all(self, images: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        corrected, intrinsics = {}, {}
        for camera, image in images.items():
            result = self.process(camera, image)
            corrected[camera] = result.image
            intrinsics[camera] = result.intrinsic
        return corrected, intrinsics
