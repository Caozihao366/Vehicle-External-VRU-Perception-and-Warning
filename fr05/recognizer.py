"""FR-05 统一异常行为识别阶段。"""

from __future__ import annotations

import math
import time

import numpy as np

from .behavior_analyzer import BehaviorAnalyzer
from .interfaces import FR05Event, FR05Result, validate_track
from .wrong_way_adapter import WrongWayAdapter


class FR05Recognizer:
    """统一识别奔跑、跌倒/蹲下和逆行，并就地更新主流程 tracks。"""

    def __init__(
        self,
        *,
        enable_wrong_way: bool = True,
        enable_lane: bool = True,
        device: str = "auto",
        orient_weights: str | None = None,
        yolopv2_weights: str | None = None,
    ):
        self.behavior_analyzer = BehaviorAnalyzer()
        self.wrong_way = (
            WrongWayAdapter(
                device=device,
                enable_lane=enable_lane,
                orient_weights=orient_weights,
                yolopv2_weights=yolopv2_weights,
            )
            if enable_wrong_way
            else None
        )

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: list,
        dt: float,
        frame_idx: int,
        *,
        timestamp: float | None = None,
        sensor_status: str = "normal",
    ) -> FR05Result:
        """执行全部 FR-05 判定，返回轨迹、事件和可观测运行状态。"""
        started = time.perf_counter()
        timestamp = time.time() if timestamp is None else float(timestamp)
        warnings: list[str] = []

        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame 必须是 HxWxC 的 numpy.ndarray")
        if not isinstance(tracks, list):
            raise TypeError("tracks 必须是 list[dict]")
        if not math.isfinite(dt) or dt <= 0:
            warnings.append(f"invalid_dt:{dt}; fallback=0.033333")
            dt = 1.0 / 30.0

        frame_h, frame_w = frame.shape[:2]
        valid_tracks = []
        invalid_track_ids = []
        for index, obj in enumerate(tracks):
            track_id = obj.get("track_id", f"index:{index}") if isinstance(obj, dict) else f"index:{index}"
            errors = validate_track(obj) if isinstance(obj, dict) else ["track must be dict"]
            if errors:
                invalid_track_ids.append(track_id)
                warnings.append(f"invalid_track[{track_id}]:" + ",".join(errors))
                if isinstance(obj, dict):
                    obj["behavior"] = "uncertain"
                    obj["behavior_conf"] = 0.0
                    obj["behavior_source"] = "interface_validation"
                continue
            valid_tracks.append(obj)

        BehaviorAnalyzer.bind_rider_to_vehicle(valid_tracks)
        for obj in valid_tracks:
            if obj.get("is_rider", False):
                obj["behavior"] = "normal"
                obj["behavior_conf"] = 1.0
                obj["behavior_source"] = "rider_vehicle_binding"
            elif obj.get("cls") in (0, 1, 3):
                behavior, confidence = self.behavior_analyzer.analyze(
                    obj, dt, frame_h, frame_w
                )
                obj["behavior"] = behavior
                obj["behavior_conf"] = confidence
                obj["behavior_source"] = (
                    "motion_pose" if behavior != "normal" else "none"
                )
            else:
                obj["behavior"] = "normal"
                obj["behavior_conf"] = 1.0
                obj["behavior_source"] = "none"

        wrong_way_diag = None
        if self.wrong_way is not None:
            try:
                wrong_way_diag = self.wrong_way.process_frame(
                    frame, valid_tracks, frame_idx
                )
                warnings.extend(wrong_way_diag.warnings)
            except Exception as exc:
                warnings.append(f"wrong_way_stage_failed:{exc}")

        tracks.extend(self.behavior_analyzer.get_active_ghosts())
        events = []
        for obj in tracks:
            obj["behavior_frame_id"] = int(frame_idx)
            obj["behavior_timestamp"] = timestamp
            state = obj.get("behavior", "normal")
            if state == "normal":
                continue
            bbox = tuple(int(value) for value in obj.get("bbox", (0, 0, 0, 0)))
            events.append(
                FR05Event(
                    frame_id=int(frame_idx),
                    timestamp=timestamp,
                    object_id=obj.get("track_id", "unknown"),
                    class_id=int(obj.get("cls", -1)),
                    state=state,
                    confidence=float(obj.get("behavior_conf", 0.0)),
                    source=obj.get("behavior_source", "unknown"),
                    bbox_px=bbox,
                    is_ghost=bool(obj.get("is_ghost", False)),
                    orient_face=obj.get("orient_face"),
                    orient_confidence=obj.get("orient_conf"),
                    ego_lane_relation=obj.get("ego_relation"),
                    decision_source=obj.get("decision_source"),
                )
            )

        if sensor_status in {"degraded", "partial", "unavailable", "unknown"}:
            warnings.append(f"upstream_sensor_status:{sensor_status}")

        capabilities = {
            "running_detection": True,
            "falling_detection": True,
            "wrong_way_detection": self.wrong_way is not None,
            "orientation_classifier": bool(
                wrong_way_diag and wrong_way_diag.orientation_enabled
            ),
            "lane_segmentation": bool(
                wrong_way_diag and wrong_way_diag.lane_enabled
            ),
            "lane_result_ready": bool(
                wrong_way_diag and wrong_way_diag.lane_ready
            ),
        }
        if tracks and not valid_tracks:
            status = "unavailable"
        elif warnings:
            status = "degraded"
        else:
            status = "available"

        return FR05Result(
            frame_id=int(frame_idx),
            timestamp=timestamp,
            tracks=tracks,
            events=events,
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            capabilities=capabilities,
            warnings=list(dict.fromkeys(warnings)),
            invalid_track_ids=invalid_track_ids,
        )
