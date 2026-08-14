"""
FR05 逆行检测 — 总线接入 Pipeline

用法:
    from fr05_wrongway import WrongWayPipeline

    pipe = WrongWayPipeline(device="auto")
    out = pipe.process_frame(frame_bgr)
    for obj in out["objects"]:
        if obj["behavior"] == "wrong_way":
            ...
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ultralytics import YOLO

from . import config as cfg
from .device_utils import resolve_device_str
from .vehicle_orientation import VehicleOrientationClassifier
from .vru_tracker import VRUTracker
from .wrong_way import WrongWayDetector
from .yolopv2_lane import YOLOPv2Lane, estimate_lane_heading_deg, overlay_lane


def _obj_to_bus_dict(obj: dict) -> dict[str, Any]:
    x1, y1, x2, y2 = obj["bbox"]
    bx, by = obj["bottom_center"]
    return {
        "track_id": int(obj["track_id"]),
        "class_id": int(obj["cls"]),
        "class_name": obj.get("class_name", str(obj["cls"])),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "bottom_center": [int(bx), int(by)],
        "behavior": obj.get("behavior", "normal"),
        "behavior_conf": float(obj.get("behavior_conf", 0.0)),
        "orient_face": obj.get("orient_face", "unknown"),
        "orient_conf": float(obj.get("orient_conf", 0.0)),
        "ego_relation": obj.get("ego_relation", "unknown"),
        "decision_source": obj.get("decision_source", "none"),
        "speed_norm": float(obj.get("speed_norm", 0.0)),
        "heading_deg": obj.get("heading_deg"),
    }


@dataclass
class WrongWayPipeline:
    """FR05 逆行检测总线模块：检测 → 跟踪 → 朝向 → 车道 → 逆行判定。"""

    device: str = cfg.DEVICE
    conf: float = cfg.CONF_THRESHOLD
    enable_yolop: bool = cfg.ENABLE_YOLOPV2
    draw_lane: bool = cfg.DRAW_LANE
    draw_ego_lane: bool = cfg.DRAW_EGO_LANE

    det_model_path: str = cfg.DET_MODEL_PATH
    orient_weights: str = cfg.ORIENT_WEIGHTS
    yolop_weights: str = cfg.YOLOPV2_WEIGHTS

    _det_device: str = field(init=False, repr=False)
    _yolop_device: str = field(init=False, repr=False)
    _orient_device: str = field(init=False, repr=False)
    _model: YOLO = field(init=False, repr=False)
    _tracker: VRUTracker = field(init=False, repr=False)
    _wrong_way: WrongWayDetector = field(init=False, repr=False)
    _orient: VehicleOrientationClassifier = field(init=False, repr=False)
    _lane_model: YOLOPv2Lane | None = field(init=False, repr=False, default=None)

    _frame_idx: int = field(init=False, default=0)
    _prev_time: float = field(init=False, default=0.0)
    _fps: float = field(init=False, default=0.0)
    _orient_cache: dict = field(init=False, default_factory=dict)
    _lane_mask: np.ndarray | None = field(init=False, default=None)
    _da_mask: np.ndarray | None = field(init=False, default=None)
    _lane_heading: float | None = field(init=False, default=None)

    def __post_init__(self):
        self._det_device = resolve_device_str(self.device)
        self._yolop_device = resolve_device_str(cfg.YOLOPV2_DEVICE or self.device)
        self._orient_device = resolve_device_str(cfg.ORIENT_DEVICE or self.device)

        self._model = YOLO(self.det_model_path)
        self._tracker = VRUTracker()
        self._wrong_way = WrongWayDetector()
        self._orient = VehicleOrientationClassifier(
            self.orient_weights,
            device=self._orient_device,
            img_size=cfg.ORIENT_IMG_SIZE,
            conf_thresh=cfg.ORIENT_CONF_THRESH,
        )
        if self.enable_yolop:
            self._lane_model = YOLOPv2Lane(
                self.yolop_weights,
                device=self._yolop_device,
                img_size=cfg.YOLOPV2_IMG_SIZE,
            )
        self._prev_time = time.time()

    def reset(self):
        """新视频流/新会话时重置内部状态。"""
        self._frame_idx = 0
        self._prev_time = time.time()
        self._fps = 0.0
        self._orient_cache.clear()
        self._lane_mask = None
        self._da_mask = None
        self._lane_heading = None
        self._tracker = VRUTracker()
        self._wrong_way = WrongWayDetector()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        annotate: bool = False,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """处理单帧 BGR 图像，返回总线结构化结果。"""
        self._frame_idx += 1
        now = timestamp if timestamp is not None else time.time()
        dt = max(now - self._prev_time, 1e-3)
        self._prev_time = now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps > 0 else 1.0 / dt

        results = self._model.track(
            frame_bgr,
            device=self._det_device,
            classes=cfg.TARGET_CLASSES,
            conf=self.conf,
            tracker=cfg.TRACKER_CFG,
            verbose=False,
            persist=True,
        )

        boxes_data = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clses = results[0].boxes.cls.cpu().numpy().astype(int)
            for box, track_id, cls in zip(boxes, ids, clses):
                x1, y1, x2, y2 = map(int, box)
                boxes_data.append((x1, y1, x2, y2, track_id, int(cls)))

        vru_results = self._tracker.update(boxes_data, dt)
        vehicle_set = set(cfg.VEHICLE_ORIENT_CLASSES)
        active_ids = set()

        for obj in vru_results:
            tid = obj["track_id"]
            active_ids.add(tid)
            if obj["cls"] not in vehicle_set:
                obj["orient_face"] = "n/a"
                obj["orient_conf"] = 0.0
                continue
            need = (
                self._orient.enabled
                and (tid not in self._orient_cache or self._frame_idx % cfg.ORIENT_EVERY_N == 0)
            )
            if need:
                face, conf, _ = self._orient.predict_bbox(frame_bgr, obj["bbox"])
                self._orient_cache[tid] = (face, conf)
            face, conf = self._orient_cache.get(tid, ("unknown", 0.0))
            obj["orient_face"] = face
            obj["orient_conf"] = conf

        for tid in list(self._orient_cache.keys()):
            if tid not in active_ids:
                del self._orient_cache[tid]

        if self._lane_model is not None and (self._frame_idx % cfg.YOLOPV2_EVERY_N == 0):
            self._lane_mask, self._da_mask = self._lane_model.infer(frame_bgr)
            self._lane_heading = estimate_lane_heading_deg(self._lane_mask)

        if self._lane_mask is not None:
            self._wrong_way.update(vru_results, self._lane_mask, self._da_mask)

        bus_objects = [_obj_to_bus_dict(o) for o in vru_results]
        wrong_way_events = [o for o in bus_objects if o["behavior"] == "wrong_way"]

        out: dict[str, Any] = {
            "frame_idx": self._frame_idx,
            "fps": round(self._fps, 2),
            "objects": bus_objects,
            "wrong_way_events": wrong_way_events,
            "wrong_way_count": len(wrong_way_events),
            "ego_lane_source": self._wrong_way.last_ego_source,
            "lane_heading_deg": self._lane_heading,
            "device": self._det_device,
        }

        if annotate:
            ego_mask = self._wrong_way.last_ego_mask
            vis = frame_bgr
            if self.draw_lane and self._lane_mask is not None:
                vis = overlay_lane(
                    vis,
                    self._lane_mask,
                    self._da_mask,
                    ego_mask=ego_mask if self.draw_ego_lane else None,
                )
            out["annotated_frame"] = vis

        return out

    @property
    def info(self) -> dict[str, Any]:
        return {
            "module": "FR05_wrong_way",
            "det_model": self.det_model_path,
            "device": self._det_device,
            "yolop_enabled": self._lane_model is not None,
            "orient_enabled": self._orient.enabled,
        }
