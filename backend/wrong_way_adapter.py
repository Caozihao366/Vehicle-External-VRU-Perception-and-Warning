"""
逆行检测适配器
==============
封装 fr05_wrongway 模块，消费统一检测器的跟踪结果，
不再内部重复跑 YOLO 检测和 ByteTrack。

职责:
  1. 对机动车 (car/bus/truck/motorcycle) 做朝向分类 (front/rear)
  2. 可选跑 YOLOPv2 车道线分割
  3. WrongWayDetector 判定逆行
  4. 将逆行行为合并回统一 tracks
"""
import numpy as np

from fr05_wrongway.wrong_way import WrongWayDetector
from fr05_wrongway.vehicle_orientation import VehicleOrientationClassifier
from fr05_wrongway.config import (
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
        from system_config import (
            FR05_ORIENT_WEIGHTS,
            FR05_YOLOPV2_WEIGHTS,
        )

        self.orient_every_n = orient_every_n
        self.lane_every_n = lane_every_n
        self.vehicle_classes = set(VEHICLE_ORIENT_CLASSES)

        # 朝向分类器
        try:
            self.orient_clf = VehicleOrientationClassifier(
                weights_path=str(orient_weights or FR05_ORIENT_WEIGHTS),
                device=device,
            )
        except Exception as e:
            print(f"[wrong_way] 朝向分类器加载失败，逆行检测将受限: {e}")
            self.orient_clf = None

        # 车道线分割（可选，模型较重）
        self.lane_model = None
        if enable_lane:
            try:
                from fr05_wrongway.yolopv2_lane import YOLOPv2Lane
                self.lane_model = YOLOPv2Lane(
                    weights=str(yolopv2_weights or FR05_YOLOPV2_WEIGHTS),
                    device=device,
                )
            except (FileNotFoundError, Exception) as e:
                print(f"[wrong_way] YOLOPv2 不可用，退化为无车道线模式: {e}")
                self.lane_model = None

        self.detector = WrongWayDetector()

        # 缓存：上一次的朝向结果（隔帧推理时保持）
        self._orient_cache = {}
        self._lane_mask = None
        self._da_mask = None

    def process_frame(self, frame: np.ndarray, tracks: list, frame_idx: int = 0):
        """
        处理一帧，就地更新 tracks 中机动车的 behavior。

        Args:
            frame: BGR 图像
            tracks: vru_perception 输出的 track dict 列表
            frame_idx: 帧序号
        """
        h, w = frame.shape[:2]

        # ---- 1. 车道线分割（隔帧） ----
        if self.lane_model is not None and frame_idx % self.lane_every_n == 0:
            try:
                self._lane_mask, self._da_mask = self.lane_model.infer(frame)
            except Exception as e:
                print(f"[wrong_way] 车道线推理失败: {e}")
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
                    except Exception:
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
            merged = self._merge_behavior(vru_behavior, ww_behavior)
            obj["behavior"] = merged
            if merged == ww_behavior:
                obj["behavior_conf"] = ww_conf

        # 清理消失目标的朝向缓存
        active_ids = {t["track_id"] for t in vehicle_tracks}
        for tid in list(self._orient_cache.keys()):
            if tid not in active_ids:
                self._orient_cache.pop(tid, None)

    @staticmethod
    def _merge_behavior(old: str, new: str) -> str:
        """按优先级合并行为，保留更紧急的。"""
        p_old = _BEHAVIOR_PRIORITY.get(old, 0)
        p_new = _BEHAVIOR_PRIORITY.get(new, 0)
        return new if p_new > p_old else old
