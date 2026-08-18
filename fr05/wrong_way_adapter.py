"""
逆行检测适配器
==============
消费主流程统一检测器的跟踪结果，不再内部重复运行 YOLO 和 ByteTrack。

职责:
  1. 对机动车 (car/bus/truck/motorcycle) 做朝向分类 (front/rear)
  2. 可选跑 YOLOPv2 车道线分割
  3. WrongWayDetector 判定逆行
  4. 将逆行行为合并回统一 tracks
"""
from pathlib import Path

import numpy as np

from .interfaces import WrongWayDiagnostics
from .wrong_way import WrongWayDetector
from .vehicle_orientation import VehicleOrientationClassifier
from .config import (
    VEHICLE_ORIENT_CLASSES,
    ORIENT_EVERY_N,
    YOLOPV2_EVERY_N,
)

# 行为优先级（数值越大越紧急）
_BEHAVIOR_PRIORITY = {
    "normal": 0,
    "suspect_wrong_way": 1,
    "running": 2,
    "wrong_way": 3,
    "falling": 4,
}


class WrongWayAdapter:
    """
    用法:
        ww = WrongWayAdapter()
        ww.process_frame(frame, tracks, frame_idx)
        # tracks 中机动车目标的 behavior 被更新
    """

    def __init__(
        self,
        orient_weights: str = None,
        yolopv2_weights: str = None,
        device: str = "cuda",
        enable_lane: bool = True,
        orient_every_n: int = ORIENT_EVERY_N,
        lane_every_n: int = YOLOPV2_EVERY_N,
    ):
        package_root = Path(__file__).resolve().parent
        orient_weights = orient_weights or package_root / "weights" / "vehicle_orientation.pt"
        yolopv2_weights = (
            yolopv2_weights
            or package_root / "third_party" / "YOLOPv2" / "data" / "weights" / "yolopv2.pt"
        )

        self.orient_every_n = orient_every_n
        self.lane_every_n = lane_every_n
        self.vehicle_classes = set(VEHICLE_ORIENT_CLASSES)
        self._init_warnings = []

        # 朝向分类器
        try:
            self.orient_clf = VehicleOrientationClassifier(
                weights_path=str(orient_weights),
                device=device,
            )
        except Exception as e:
            self._init_warnings.append(f"orientation_unavailable: {e}")
            self.orient_clf = None

        # 车道线分割（可选，模型较重）
        self.lane_model = None
        if enable_lane:
            try:
                from .yolopv2_lane import YOLOPv2Lane
                self.lane_model = YOLOPv2Lane(
                    weights=str(yolopv2_weights),
                    device=device,
                )
            except Exception as e:
                self._init_warnings.append(f"lane_unavailable: {e}")
                self.lane_model = None

        self.detector = WrongWayDetector()

        # 缓存：上一次的朝向结果（隔帧推理时保持）
        self._orient_cache = {}
        self._lane_mask = None
        self._da_mask = None

    def process_frame(
        self, frame: np.ndarray, tracks: list, frame_idx: int = 0
    ) -> WrongWayDiagnostics:
        """
        处理一帧，就地更新 tracks 中机动车的 behavior。

        Args:
            frame: BGR 图像
            tracks: vru_perception 输出的 track dict 列表
            frame_idx: 帧序号
        """
        warnings = list(self._init_warnings)

        # ---- 1. 车道线分割（隔帧） ----
        if self.lane_model is not None and frame_idx % self.lane_every_n == 0:
            try:
                self._lane_mask, self._da_mask = self.lane_model.infer(frame)
            except Exception as e:
                warnings.append(f"lane_inference_failed: {e}")
                self._lane_mask = None
                self._da_mask = None

        # ---- 2. 筛选机动车 track ----
        vehicle_tracks = [t for t in tracks if t.get("cls") in self.vehicle_classes]

        # ---- 3. 朝向分类（隔帧） ----
        do_orient = (frame_idx % self.orient_every_n == 0)
        for obj in vehicle_tracks:
            tid = obj["track_id"]
            if self.orient_clf is not None and (do_orient or tid not in self._orient_cache):
                bbox = obj.get("bbox")
                if bbox is not None:
                    try:
                        face, conf, _ = self.orient_clf.predict_bbox(frame, bbox)
                        self._orient_cache[tid] = (face, conf)
                    except Exception as exc:
                        warnings.append(f"orientation_inference_failed[{tid}]: {exc}")
                        face, conf = "unknown", 0.0
                        self._orient_cache[tid] = (face, conf)
            face, conf = self._orient_cache.get(tid, ("unknown", 0.0))
            obj["orient_face"] = face
            obj["orient_conf"] = conf

        # ---- 4. 逆行判定 ----
        # WrongWayDetector.update() 会重置传入 track 的 behavior，
        # 所以先保存 vru 模块已判定的行为，只传机动车 track。
        for obj in vehicle_tracks:
            obj["_vru_behavior"] = obj.get("behavior", "normal")
            obj["_vru_behavior_conf"] = obj.get("behavior_conf", 0.0)
            obj["_vru_behavior_source"] = obj.get("behavior_source", "none")

        if vehicle_tracks:
            self.detector.update(
                vehicle_tracks,
                lane_mask=self._lane_mask,
                da_mask=self._da_mask,
            )

        # ---- 5. 行为合并（按优先级） ----
        for obj in vehicle_tracks:
            ww_behavior = obj.get("behavior", "normal")
            ww_conf = obj.get("behavior_conf", 0.0)
            vru_behavior = obj.pop("_vru_behavior", "normal")
            vru_conf = obj.pop("_vru_behavior_conf", 0.0)
            vru_source = obj.pop("_vru_behavior_source", "none")
            merged = self._merge_behavior(vru_behavior, ww_behavior)
            obj["behavior"] = merged
            if merged == ww_behavior:
                obj["behavior_conf"] = ww_conf
                obj["behavior_source"] = (
                    "wrong_way_fusion" if ww_behavior != "normal" else "none"
                )
            else:
                obj["behavior_conf"] = vru_conf
                obj["behavior_source"] = vru_source

        # 清理消失目标的朝向缓存
        active_ids = {t["track_id"] for t in vehicle_tracks}
        for tid in list(self._orient_cache.keys()):
            if tid not in active_ids:
                self._orient_cache.pop(tid, None)

        return WrongWayDiagnostics(
            orientation_enabled=bool(
                self.orient_clf is not None and self.orient_clf.enabled
            ),
            lane_enabled=self.lane_model is not None,
            lane_ready=self._lane_mask is not None,
            vehicle_count=len(vehicle_tracks),
            wrong_way_count=sum(
                obj.get("behavior") == "wrong_way" for obj in vehicle_tracks
            ),
            suspect_count=sum(
                obj.get("behavior") == "suspect_wrong_way"
                for obj in vehicle_tracks
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _merge_behavior(old: str, new: str) -> str:
        """按优先级合并行为，保留更紧急的。"""
        p_old = _BEHAVIOR_PRIORITY.get(old, 0)
        p_new = _BEHAVIOR_PRIORITY.get(new, 0)
        return new if p_new > p_old else old
