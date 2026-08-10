#!/usr/bin/env python3
"""独立 demo：python -m fr05_wrongway.demo --source videos/test.mp4"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# 允许从仓库根目录运行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fr05_wrongway import WrongWayPipeline
from fr05_wrongway import config as cfg


def draw_overlay(frame, out):
    vis = frame.copy()
    for obj in out["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        behavior = obj["behavior"]
        if behavior == "wrong_way":
            color = (0, 0, 255)
        elif behavior == "suspect_wrong_way":
            color = (0, 165, 255)
        else:
            color = (200, 200, 200)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = (
            f"ID{obj['track_id']} {obj['class_name']} {behavior} "
            f"{obj['orient_face']}:{obj['orient_conf']:.2f} ego={obj['ego_relation']}"
        )
        cv2.putText(vis, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(
        vis, f"FPS {out['fps']:.1f} | wrong {out['wrong_way_count']}",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )
    return vis


def main():
    p = argparse.ArgumentParser(description="FR05 wrong-way demo")
    p.add_argument("--source", default="0")
    p.add_argument("--device", default=cfg.DEVICE)
    p.add_argument("--conf", type=float, default=cfg.CONF_THRESHOLD)
    p.add_argument("--no-yolop", action="store_true")
    args = p.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    pipe = WrongWayPipeline(
        device=args.device,
        conf=args.conf,
        enable_yolop=not args.no_yolop,
    )
    print("[FR05]", pipe.info)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开: {source}")

    cv2.namedWindow(cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = pipe.process_frame(frame, annotate=True)
        vis = out.get("annotated_frame", frame)
        vis = draw_overlay(vis, out)
        cv2.imshow(cfg.WINDOW_NAME, vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
        if cv2.getWindowProperty(cfg.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
