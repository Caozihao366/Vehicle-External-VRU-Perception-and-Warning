"""Simple CPU detection script using Ultralytics YOLOv8.

Usage: python detect.py --input input --output output/detections --max-frames 1

This will process synchronized frames (default 1), run detection on each camera
image using `yolov8n` on CPU, save boxed images and a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from ultralytics import YOLO

from bev映射.bev import synchronized_frames, load_calibrations, BEVProjector
from bev映射.yolo_detector import process_images

#把YOLO检测结果画到原图上
def draw_boxes(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, names) -> np.ndarray:
    out = image.copy()
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        color = (0, 255, 0)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{names[int(c)]} {s:.2f}"
        cv2.putText(out, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

#通过坐标转换，把yolo和bev连接
def backproject_to_ground(calib, u: float, v: float) -> tuple[float, float, float]:
    """输入一个图片上的像素点(u,v)
    利用摄像头标定参数K、R、T
    把它沿着摄像头光线投射到地面平面Z=0上
    得到它在车辆坐标系中的真实位置(x,y,0)
    """
    #读取K
    K = calib.intrinsic
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 像素 → 相机射线
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)

    # 相机坐标 → ego坐标
    R = calib.rotation
    R_inv = R.T
    d_ego = R_inv @ d_cam

    # 获取相机在ego坐标系中的位置
    cam_pos = calib.translation.astype(np.float64)

    # 求射线与地面平面 z=0 的交点
    if abs(d_ego[2]) < 1e-8:
        return (float('nan'), float('nan'), 0.0)
    alpha = -cam_pos[2] / d_ego[2]
    ground = cam_pos + alpha * d_ego
    return float(ground[0]), float(ground[1]), 0.0

def build_detection_payload(det: dict, gx: float, gy: float, gz: float) -> dict:
    """Return a detection payload that stores ground-plane coordinates instead of distance."""
    payload = dict(det)
    payload["coordinate_m"] = [float(gx), float(gy)]
    payload["ground_point_m"] = [float(gx), float(gy), float(gz)]
    payload.pop("distance_m", None)
    return payload


#显示坐标（米）转换成bev像素
def ground_to_bev_pixel(x: float, y: float, projector: BEVProjector) -> tuple[int, int]:
    i = int(round((projector.x_max - x) / projector.resolution - 0.5))
    j = int(round((projector.y_max - y) / projector.resolution - 0.5))
    # clamp
    i = max(0, min(projector.height - 1, i))
    j = max(0, min(projector.width - 1, j))
    return i, j

#参数配置入口
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("input"))
    p.add_argument("--output", type=Path, default=Path("output/detections"))
    p.add_argument("--start-frame", type=int, default=0, help="Zero-based frame index at which to start processing")
    p.add_argument("--max-frames", type=int, default=1)
    p.add_argument("--tolerance-ms", type=float, default=60.0)
    p.add_argument("--calibration", type=Path, default=Path("calibrated_sensor.json"))
    p.add_argument("--sensors", type=Path, default=Path("sensor.json"))
    p.add_argument("--x-range", nargs=2, type=float, default=(-10.0, 30.0))
    p.add_argument("--y-range", nargs=2, type=float, default=(-15.0, 15.0))
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # load small model (yolov8n). Ultralytics will download weights if absent.
    model = YOLO('yolov8n.pt')

    frames = synchronized_frames(args.input, None, args.tolerance_ms)
    summary = {}
    coordinate_export = []
    start = time.time()
    projector = None
    calibrations = None
    for idx, (scene, ts, paths) in enumerate(frames):
        if idx < args.start_frame:
            continue
        if idx - args.start_frame >= args.max_frames:
            break
        frame_dir = args.output / scene / f"frame_{idx:04d}_{ts}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        # read images into memory
        images = {camera: cv2.imread(str(p)) for camera, p in paths.items()}
        unreadable = [c for c, im in images.items() if im is None]
        if unreadable:
            raise OSError(f"Could not read: {', '.join(unreadable)}")

        # init calibrations and projector once
        if projector is None:
            calibrations = load_calibrations(args.calibration, args.sensors, scene)
            projector = BEVProjector(calibrations, *args.x_range, *args.y_range, args.resolution)

        t0 = time.time()
        results = process_images(model, images, paths, frame_dir, draw_boxes, imgsz=args.imgsz)

        # generate BEV and overlay detections
        bev = projector.project(images)
        # for each detection, backproject bottom-center and draw on BEV
        for cam, dets in results.items():
            calib = calibrations[cam]
            for det in dets:
                x1, y1, x2, y2 = det['bbox']
                u = (x1 + x2) / 2.0
                v = y2
                gx, gy, gz = backproject_to_ground(calib, u, v)
                if np.isnan(gx) or np.isnan(gy):
                    det['coordinate_m'] = None
                    det['ground_point_m'] = None
                    continue
                det.update(build_detection_payload(det, gx, gy, gz))
                i, j = ground_to_bev_pixel(gx, gy, projector)
                # draw circle and label with coordinates
                cv2.circle(bev, (j, i), 4, (0, 0, 255), -1)
                label = f"{det['label']} ({gx:.2f}, {gy:.2f})"
                cv2.putText(bev, label, (j + 6, i - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        bev_path = frame_dir / f"bev_{idx:04d}_{ts}.jpg"
        cv2.imwrite(str(bev_path), bev)

        t1 = time.time()
        frame_record = {
            "scene": scene,
            "timestamp": int(ts),
            "detections": results,
            "bev": str(bev_path),
            "frame_dir": str(frame_dir),
            "elapsed_s": t1 - t0,
        }
        summary[f"frame_{idx}"] = frame_record
        for cam, dets in results.items():
            for det in dets:
                if det.get("coordinate_m") is None:
                    continue
                coordinate_export.append({
                    "scene": scene,
                    "timestamp": int(ts),
                    "camera": cam,
                    "label": det.get("label"),
                    "coordinate_m": det.get("coordinate_m"),
                    "ground_point_m": det.get("ground_point_m"),
                    "bbox": det.get("bbox"),
                    "score": det.get("score"),
                })
        print(f"Processed frame {idx} scene={scene} time={t1 - t0:.2f}s -> {bev_path}")

    total = time.time() - start
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    coordinates_path = args.output / "coordinates.json"
    coordinates_path.write_text(json.dumps(coordinate_export, indent=2), encoding='utf-8')
    print(f"Done. Processed {len(summary)} frames in {total:.2f}s. Results: {summary_path}")
    print(f"Coordinate export: {coordinates_path}")


if __name__ == '__main__':
    main()
