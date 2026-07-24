from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np
from ultralytics import YOLO


def process_images(
    model: YOLO,
    images: Dict[str, np.ndarray],
    paths: Dict[str, Path],
    out_dir: Path,
    draw_func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict], np.ndarray],
    imgsz: int = 640,
) -> Dict[str, list]:
    results_summary: Dict[str, list] = {}
    for cam, img in images.items():
        if img is None:
            continue
        res = model.predict(source=img, device='cpu', imgsz=imgsz, verbose=False)[0]
        if hasattr(res, 'boxes') and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, 'cpu') else np.asarray(res.boxes.xyxy)
            scores = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, 'cpu') else np.asarray(res.boxes.conf)
            classes = res.boxes.cls.cpu().numpy() if hasattr(res.boxes.cls, 'cpu') else np.asarray(res.boxes.cls)
        else:
            boxes = np.zeros((0, 4))
            scores = np.zeros((0,))
            classes = np.zeros((0,))

        names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else {i: str(i) for i in range(80)}
        out_name = paths[cam].name if cam in paths else f"{cam}.jpg"
        out_img = draw_func(img, boxes, scores, classes, names)
        out_path = out_dir / f"{cam}_{out_name}"
        cv2.imwrite(str(out_path), out_img)

        vru_labels = {"person", "bicycle", "motorcycle", "bike", "cyclist", "pedestrian", "rider"}
        detections = []
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
            label = names[int(c)]
            if label.lower() not in vru_labels:
                continue
            detections.append({
                "class": int(c),
                "label": label,
                "score": float(s),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "coordinate_m": None,
                "ground_point_m": None,
            })
        results_summary[cam] = detections
    return results_summary
