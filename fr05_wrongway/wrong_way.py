"""逆行判定（机制优先，不是只调阈值）：

硬条件（同时满足才允许 wrong_way）：
  1) 目标在本车道内，或侵入本车道（YOLOPv2 车道）
  2) 机动车且朝向为 front（车头）

不在本车道：哪怕是 front，最多 suspect_wrong_way。
"""
from __future__ import annotations

from .config import (
    EGO_INVADE_OVERLAP,
    ORIENT_CONF_THRESH,
    VEHICLE_ORIENT_CLASSES,
    WRONG_WAY_CLEAR_FRAMES,
    WRONG_WAY_CONFIRM_FRAMES,
    WRONG_WAY_SCORE_DECAY,
)
from .yolopv2_lane import build_ego_lane_mask, ego_lane_relation, estimate_lane_heading_deg


class WrongWayDetector:
    def __init__(
        self,
        confirm_frames: int = WRONG_WAY_CONFIRM_FRAMES,
        clear_frames: int = WRONG_WAY_CLEAR_FRAMES,
        score_decay: float = WRONG_WAY_SCORE_DECAY,
        invade_overlap: float = EGO_INVADE_OVERLAP,
        orient_conf_thresh: float = ORIENT_CONF_THRESH,
    ):
        self.confirm_frames = confirm_frames
        self.clear_frames = clear_frames
        self.score_decay = score_decay
        self.invade_overlap = invade_overlap
        self.orient_conf_thresh = orient_conf_thresh
        self.vehicle_classes = set(VEHICLE_ORIENT_CLASSES)
        self._score = {}
        self._clear = {}
        self._latched = {}
        self.last_ego_mask = None
        self.last_ego_source = "none"

    def update(self, vru_results, lane_mask, da_mask=None):
        ego_mask = build_ego_lane_mask(lane_mask=lane_mask, da_mask=da_mask)
        self.last_ego_mask = ego_mask
        self.last_ego_source = getattr(build_ego_lane_mask, "last_source", "none")

        active = set()
        for obj in vru_results:
            tid = obj["track_id"]
            active.add(tid)

            bx, by = obj.get("bottom_center", (None, None))
            face = obj.get("orient_face", "unknown")
            face_conf = float(obj.get("orient_conf", 0.0))
            cls = obj.get("cls")
            bbox = obj.get("bbox")

            lane_heading = None
            if lane_mask is not None and bx is not None:
                lane_heading = estimate_lane_heading_deg(lane_mask, (bx, by))

            relation = ego_lane_relation(
                ego_mask, bbox, (bx, by), invade_overlap=self.invade_overlap
            )
            same_lane = relation in ("in_ego", "invade_ego")
            is_vehicle = cls in self.vehicle_classes
            is_front = (
                is_vehicle
                and face == "front"
                and face_conf >= self.orient_conf_thresh
            )

            obj["lane_heading_deg"] = lane_heading
            obj["ego_relation"] = relation
            obj["angle_diff_deg"] = None
            obj["orient_vs_lane_diff"] = None
            obj["behavior"] = "normal"
            obj["behavior_conf"] = 0.0
            obj["decision_source"] = "none"

            score = self._score.get(tid, 0.0)
            clear = self._clear.get(tid, 0)
            latched = self._latched.get(tid, False)

            # ---- 证据：只有「同车道 + 车头」才给确认向的强分 ----
            if same_lane and is_front:
                score = min(float(self.confirm_frames), score + 1.0)
                clear = 0
                obj["decision_source"] = "same_lane+front"
            elif is_front and not same_lane:
                # 对向车道见车头：正常现象，最多轻微嫌疑，绝不确认
                score = min(float(self.confirm_frames) * 0.6, score + 0.35)
                clear = 0
                obj["decision_source"] = "front_but_other_lane"
            else:
                score = max(0.0, score - self.score_decay)
                clear += 1

            # 硬门槛：不在同车道，禁止锁存 wrong_way
            if same_lane and is_front and score >= self.confirm_frames:
                latched = True
            if not same_lane:
                latched = False
            if latched and clear >= self.clear_frames:
                latched = False
                score = 0.0
                clear = 0

            self._score[tid] = score
            self._clear[tid] = clear
            self._latched[tid] = latched

            if latched and same_lane and is_front:
                obj["behavior"] = "wrong_way"
                obj["behavior_conf"] = min(1.0, score / max(1, self.confirm_frames))
            elif score > 0:
                obj["behavior"] = "suspect_wrong_way"
                obj["behavior_conf"] = score / max(1, self.confirm_frames)

        for tid in list(self._score.keys()):
            if tid not in active:
                self._score.pop(tid, None)
                self._clear.pop(tid, None)
                self._latched.pop(tid, None)

        return vru_results
