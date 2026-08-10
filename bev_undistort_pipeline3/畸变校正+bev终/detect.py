"""End-to-end: synchronize -> undistort -> detect -> BEV -> export."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from bev import BEVProjector, load_calibrations, synchronized_frames, with_intrinsics
from undistortion import Undistorter, load_distortions
from yolo_detector import detect_images

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(r"D:\test2\a2d2-preview")
DEFAULT_WEIGHTS = Path(r"D:\test2\bev_new2\yolov8n.pt")

def backproject_to_ground(calib, u, v):
    ray_camera = np.linalg.inv(calib.intrinsic) @ np.array([u, v, 1.0], dtype=np.float64)
    ray_ego = calib.rotation.T @ ray_camera
    if abs(ray_ego[2]) < 1e-8:
        return None
    scale = -calib.translation[2]/ray_ego[2]
    if scale <= 0:
        return None
    point = calib.translation + scale*ray_ego
    return float(point[0]), float(point[1]), 0.0


def draw_detection(image, detection, color=(0, 220, 255)):
    x1, y1, x2, y2 = map(int, detection["bbox"])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f'{detection["label"]} {detection["score"]:.2f}', (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)


def parse_args():
    p = argparse.ArgumentParser(description="A2D2 undistortion + YOLO + six-camera BEV")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--calibration", type=Path, help="A2D2 cams_lidars.json; defaults to INPUT/cams_lidars.json")
    p.add_argument("--output", type=Path, default=PROJECT_DIR / "output_a2d2")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS); p.add_argument("--scene")
    p.add_argument("--max-frames", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--confidence", type=float, default=.25)
    p.add_argument("--x-range", nargs=2, type=float, default=(-10., 30.)); p.add_argument("--y-range", nargs=2, type=float, default=(-15., 15.))
    p.add_argument("--resolution", type=float, default=.05)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"A2D2 dataset not found: {args.input}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {args.weights}")
    args.output.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    calibration_path = args.calibration or args.input / "cams_lidars.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"A2D2 calibration not found: {calibration_path}")
    undistorter = Undistorter(load_distortions(calibration_path)); model = YOLO(args.weights)
    summary = []
    base_calibrations = load_calibrations(calibration_path)
    for index, (scene, ts, paths) in enumerate(synchronized_frames(args.input, args.scene)):
        if index >= args.max_frames: break
        raw = {c: cv2.imread(str(p)) for c, p in paths.items()}
        bad = [c for c, image in raw.items() if image is None]
        if bad: raise OSError(f"Unreadable images: {', '.join(bad)}")

        #它保证畸变校正模块不仅输出新图片，还把新内参一起交给BEV和坐标计算模块。
        images, new_intrinsics = undistorter.process_all(raw)
        calibrations = with_intrinsics(base_calibrations, new_intrinsics)
        projector = BEVProjector(calibrations, *args.x_range, *args.y_range, args.resolution)
        detections = detect_images(model, images, args.imgsz, confidence=args.confidence)
        bev = projector.project(images); frame_dir = args.output/scene/f"frame_{index:04d}_{ts}"; frame_dir.mkdir(parents=True, exist_ok=True)
        for camera, items in detections.items():
            preview = images[camera].copy()
            for item in items:
                draw_detection(preview, item)
                x1, y1, x2, y2 = item["bbox"]
                point = backproject_to_ground(calibrations[camera], (x1+x2)/2., y2)
                item["ground_point_m"] = list(point) if point else None
                item["coordinate_m"] = list(point[:2]) if point else None
                if point:
                    pixel = projector.ground_to_pixel(point[0], point[1]); cv2.circle(bev, pixel, 6, (0, 220, 255), -1)
                    cv2.putText(bev, f'{item["label"]} ({point[0]:.1f},{point[1]:.1f})', (pixel[0]+7, pixel[1]-7), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,220,255), 1, cv2.LINE_AA)
            cv2.imwrite(str(frame_dir/f"{camera}_undistorted_detected.jpg"), preview)
        bev_path = frame_dir/"bev.jpg"; cv2.imwrite(str(bev_path), bev)
        summary.append({"scene": scene, "timestamp": int(ts), "bev": str(bev_path), "detections": detections})
    (args.output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processed {len(summary)} frame(s). Output: {args.output}")


if __name__ == "__main__": main()
