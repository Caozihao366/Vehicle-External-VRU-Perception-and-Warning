"""
TTC 碰撞风险引擎
================
封装 ttc 模块，支持两种模式:
  1. 匀速模型（QCNet 预热期/不可用时）
  2. QCNet 多模态轨迹预测

每目标维护 RiskStateMachine 滞回状态。
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ttc.core import compute_ttc, compute_multimodal_ttc, build_ego_trajectory
from ttc.models import EgoState as TTCEgoState, VRUState
from ttc.risk import RiskStateMachine, classify_risk
from system_config import (
    EGO_LENGTH_M,
    EGO_WIDTH_M,
    TTC_DANGER_S,
    TTC_DT_S,
    TTC_HORIZON_S,
    TTC_SAFETY_MARGIN_M,
    TTC_WARNING_S,
    VRU_RADIUS_M,
    RISK_DANGER_HOLD_FRAMES,
    RISK_DOWNGRADE_FRAMES,
)


class RiskResult:
    """单个目标的风险评估结果。"""
    __slots__ = (
        "track_id", "risk_level", "ttc_s", "ttc_ms",
        "collision_probability", "major_reason",
        "requires_aeb", "min_clearance_m", "prediction_source",
        "behavior", "position", "velocity",
    )

    def __init__(self, track_id, risk_level, ttc_s, collision_probability,
                 major_reason, requires_aeb, min_clearance_m,
                 prediction_source, behavior, position, velocity):
        self.track_id = track_id
        self.risk_level = risk_level
        self.ttc_s = ttc_s
        self.ttc_ms = int(ttc_s * 1000) if ttc_s is not None and math.isfinite(ttc_s) else None
        self.collision_probability = collision_probability
        self.major_reason = major_reason
        self.requires_aeb = requires_aeb
        self.min_clearance_m = min_clearance_m
        self.prediction_source = prediction_source
        self.behavior = behavior
        self.position = position
        self.velocity = velocity


class TTCEngine:
    """
    TTC + 风险状态机。

    用法:
        engine = TTCEngine()
        results = engine.evaluate(bev_frame, projected, qcnet_predictions=None)
    """

    def __init__(
        self,
        warning_ttc_s: float = TTC_WARNING_S,
        danger_ttc_s: float = TTC_DANGER_S,
        horizon_s: float = TTC_HORIZON_S,
        dt_s: float = TTC_DT_S,
        ego_length_m: float = EGO_LENGTH_M,
        ego_width_m: float = EGO_WIDTH_M,
        safety_margin_m: float = TTC_SAFETY_MARGIN_M,
        vru_radius_m: float = VRU_RADIUS_M,
    ):
        self.warning_ttc_s = warning_ttc_s
        self.danger_ttc_s = danger_ttc_s
        self.horizon_s = horizon_s
        self.dt_s = dt_s
        self.ego_length_m = ego_length_m
        self.ego_width_m = ego_width_m
        self.safety_margin_m = safety_margin_m
        self.vru_radius_m = vru_radius_m

        self.state_machines: Dict = {}

    def _get_sm(self, track_id) -> RiskStateMachine:
        if track_id not in self.state_machines:
            self.state_machines[track_id] = RiskStateMachine(
                downgrade_confirm_frames=RISK_DOWNGRADE_FRAMES,
                danger_hold_frames=RISK_DANGER_HOLD_FRAMES,
            )
        return self.state_machines[track_id]

    def evaluate(
        self,
        bev_frame,
        projected: list,
        qcnet_predictions: Optional[dict] = None,
        ego_speed_mps: float = 0.0,
        ego_yaw_rate_rps: float = 0.0,
    ) -> List[RiskResult]:
        """
        评估所有目标的碰撞风险。

        Args:
            bev_frame: BEVFrame（自车状态）
            projected: list[ProjectedObject]
            qcnet_predictions: dict {track_id_str: QCNetTargetPrediction}（可选）
            ego_speed_mps: 自车速度
            ego_yaw_rate_rps: 自车横摆角速度

        Returns:
            list[RiskResult]
        """
        ego = TTCEgoState(
            x=0.0, y=0.0, yaw=0.0,
            speed_mps=ego_speed_mps,
            yaw_rate_rps=ego_yaw_rate_rps,
        )

        # 自车 CTRV 轨迹（多模态 TTC 用）
        ego_traj = build_ego_trajectory(ego, horizon_s=self.horizon_s, dt_s=self.dt_s)

        results = []
        active_ids = set()

        for obj in projected:
            tid = obj.track_id
            active_ids.add(tid)
            behavior = obj.behavior if obj.behavior in (
                "normal", "running", "falling", "wrong_way", "uncertain"
            ) else "normal"

            ttc_s = None
            collision_prob = 0.0
            min_clearance = float("inf")
            pred_source = "constant_velocity"

            # ---- 尝试 QCNet 多模态 ----
            qcnet_pred = None
            if qcnet_predictions is not None:
                qcnet_pred = qcnet_predictions.get(str(tid))

            if qcnet_pred is not None and qcnet_pred.modes:
                try:
                    mm_result = compute_multimodal_ttc(
                        track_id=tid,
                        ego_trajectory=ego_traj,
                        modes=qcnet_pred.modes,
                        prediction_source=qcnet_pred.prediction_source,
                        behavior_state=behavior,
                        sensor_availability="available",
                        dt_s=self.dt_s,
                        ego_length_m=self.ego_length_m,
                        ego_width_m=self.ego_width_m,
                        safety_margin_m=self.safety_margin_m,
                        vru_radius_m=self.vru_radius_m,
                        danger_ttc_s=self.danger_ttc_s,
                        warning_ttc_s=self.warning_ttc_s,
                    )
                    if mm_result.earliest_ttc_ms is not None:
                        ttc_s = mm_result.earliest_ttc_ms / 1000.0
                    collision_prob = mm_result.collision_probability
                    min_clearance = mm_result.min_clearance_m
                    pred_source = mm_result.prediction_source
                except Exception:
                    # QCNet 多模态失败，回退匀速模型
                    qcnet_pred = None

            # ---- 匀速模型 TTC（回退） ----
            if qcnet_pred is None:
                # GroundPlaneProjector 由相邻 ego 局部坐标差分得到的速度，
                # 已经是目标相对自车的速度。这里必须把自车视为静止；
                # 否则 ego_speed 会被重复计算一次，导致 TTC 系统性偏小。
                relative_ego = TTCEgoState(
                    x=0.0, y=0.0, yaw=0.0,
                    speed_mps=0.0,
                    yaw_rate_rps=0.0,
                )
                vru = VRUState(
                    x=obj.position[0],
                    y=obj.position[1],
                    vx_mps=obj.velocity[0],
                    vy_mps=obj.velocity[1],
                    radius_m=self.vru_radius_m,
                )
                ttc_result = compute_ttc(
                    relative_ego, vru,
                    horizon_s=self.horizon_s,
                    dt_s=self.dt_s,
                    ego_length_m=self.ego_length_m,
                    ego_width_m=self.ego_width_m,
                    safety_margin_m=self.safety_margin_m,
                )
                if ttc_result.collision:
                    ttc_s = ttc_result.ttc_s
                    collision_prob = 1.0
                min_clearance = ttc_result.min_clearance_m

            # ---- 风险分级 ----
            raw_level, reason = classify_risk(
                earliest_ttc_s=ttc_s,
                collision_probability=collision_prob,
                behavior_state=behavior,
                danger_ttc_s=self.danger_ttc_s,
                warning_ttc_s=self.warning_ttc_s,
            )

            # 滞回状态机
            sm = self._get_sm(tid)
            stable_level = sm.update(tid, raw_level)

            results.append(RiskResult(
                track_id=tid,
                risk_level=stable_level,
                ttc_s=ttc_s,
                collision_probability=collision_prob,
                major_reason=reason,
                requires_aeb=(stable_level == "danger"),
                min_clearance_m=min_clearance,
                prediction_source=pred_source,
                behavior=behavior,
                position=obj.position,
                velocity=obj.velocity,
            ))

        # 清理消失目标的状态机
        for tid in list(self.state_machines.keys()):
            if tid not in active_ids:
                del self.state_machines[tid]

        return results
