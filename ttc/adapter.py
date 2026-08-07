"""JSON/dictionary boundary adapter for the TTC baseline."""

from __future__ import annotations

import math

from .core import _validate_time_series, build_ego_trajectory, compute_multimodal_ttc
from .models import EgoState, EgoTrajectoryPoint, MultimodalTTCResult, PredictionMode, VRUTrajectoryPoint


def _ego_reference_offset(ego_payload: dict) -> float:
    """Return the longitudinal offset from the supplied reference to body centre."""
    reference_point = str(ego_payload.get("reference_point", "geometric_center"))
    if reference_point == "geometric_center":
        return 0.0
    if reference_point == "rear_axle_center":
        if "rear_axle_to_center_m" not in ego_payload:
            raise ValueError(
                "rear_axle_to_center_m is required for reference_point "
                "'rear_axle_center'"
            )
        offset = float(ego_payload["rear_axle_to_center_m"])
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("rear_axle_to_center_m must be finite and non-negative")
        return offset
    raise ValueError(
        "ego.reference_point must be 'geometric_center' or 'rear_axle_center'"
    )


def _centre_ego_point(
    point: EgoTrajectoryPoint,
    longitudinal_offset_m: float,
) -> EgoTrajectoryPoint:
    return EgoTrajectoryPoint(
        t_s=point.t_s,
        x=point.x + longitudinal_offset_m * math.cos(point.yaw),
        y=point.y + longitudinal_offset_m * math.sin(point.yaw),
        yaw=point.yaw,
    )


def _payload_input_quality(payload: dict) -> str:
    """Validate optional timing metadata without burdening legacy integrations."""
    timing = payload.get("timing")
    if timing is None:
        return "valid"
    required = ("source_timestamp_ms", "evaluation_timestamp_ms")
    if any(name not in timing for name in required):
        raise ValueError(
            "timing requires source_timestamp_ms and evaluation_timestamp_ms"
        )
    source_ms = float(timing["source_timestamp_ms"])
    evaluation_ms = float(timing["evaluation_timestamp_ms"])
    max_age_ms = float(timing.get("max_input_age_ms", 250.0))
    if not all(math.isfinite(value) for value in (source_ms, evaluation_ms, max_age_ms)):
        raise ValueError("timing values must be finite")
    if max_age_ms < 0 or evaluation_ms < source_ms:
        raise ValueError("timing values are inconsistent")
    return "input_stale" if evaluation_ms - source_ms > max_age_ms else "valid"


def compute_multimodal_ttc_from_payload(payload: dict) -> MultimodalTTCResult:
    """Parse the agreed JSON boundary and compute multimodal TTC."""
    coordinate_frame = payload.get("coordinate_frame")
    if coordinate_frame != "ego_current":
        raise ValueError(
            "coordinate_frame must be 'ego_current': x forward, y left, "
            "all trajectories expressed relative to the current ego frame"
        )
    ego_payload = payload["ego"]
    reference_offset_m = _ego_reference_offset(ego_payload)
    ego_state = EgoState(
        x=float(ego_payload.get("x", 0.0)),
        y=float(ego_payload.get("y", 0.0)),
        yaw=float(ego_payload.get("yaw", 0.0)),
        speed_mps=float(ego_payload.get("speed_mps", 0.0)),
        yaw_rate_rps=float(ego_payload.get("yaw_rate_rps", 0.0)),
    )
    horizon_s = float(payload.get("ego_horizon_s", 5.0))
    dt_s = float(payload.get("dt_s", 0.1))
    external_ego_points = ego_payload.get("trajectory")
    if external_ego_points is not None:
        ego_trajectory = tuple(
            _centre_ego_point(
                EgoTrajectoryPoint(
                    t_s=float(point["t"]),
                    x=float(point["x"]),
                    y=float(point["y"]),
                    yaw=float(point["yaw"]),
                ),
                reference_offset_m,
            )
            for point in external_ego_points
        )
        _validate_time_series(ego_trajectory, "ego.trajectory")
        ego_trajectory_source = str(
            ego_payload.get("trajectory_source", "external_planner")
        )
    else:
        reference_trajectory = build_ego_trajectory(
            ego_state, horizon_s=horizon_s, dt_s=dt_s
        )
        ego_trajectory = tuple(
            _centre_ego_point(point, reference_offset_m)
            for point in reference_trajectory
        )
        ego_trajectory_source = "ctrv_fallback"

    modes = []
    for mode_payload in payload["prediction"]["modes"]:
        points = tuple(
            VRUTrajectoryPoint(
                t_s=float(point["t"]),
                x=float(point["x"]),
                y=float(point["y"]),
                sigma_x_m=float(point.get("sigma_x_m", 0.0)),
                sigma_y_m=float(point.get("sigma_y_m", 0.0)),
            )
            for point in mode_payload["points"]
        )
        modes.append(
            PredictionMode(
                mode_id=str(mode_payload["mode_id"]),
                probability=float(mode_payload["probability"]),
                points=points,
            )
        )

    vehicle = payload.get("vehicle", {})
    uncertainty = payload.get("uncertainty_config", {})
    return compute_multimodal_ttc(
        track_id=payload["track_id"],
        ego_trajectory=ego_trajectory,
        modes=modes,
        prediction_source=str(payload.get("prediction_source", "intent_model")),
        behavior_state=str(payload.get("behavior_state", "normal")),
        sensor_availability=str(payload.get("sensor_availability", "available")),
        dt_s=dt_s,
        ego_length_m=float(vehicle.get("length_m", 4.8)),
        ego_width_m=float(vehicle.get("width_m", 2.0)),
        safety_margin_m=float(vehicle.get("safety_margin_m", 0.4)),
        vru_radius_m=float(payload.get("vru_radius_m", 0.4)),
        uncertainty_base_m=float(uncertainty.get("base_m", 0.15)),
        uncertainty_growth_mps=float(uncertainty.get("growth_mps", 0.20)),
        sigma_scale=float(uncertainty.get("sigma_scale", 1.0)),
        danger_ttc_s=float(payload.get("risk_config", {}).get("danger_ttc_s", 1.2)),
        warning_ttc_s=float(
            payload.get("risk_config", {}).get("warning_ttc_s", 2.5)
        ),
        collision_check_dt_s=float(payload.get("collision_check_dt_s", 0.02)),
        ego_trajectory_source=ego_trajectory_source,
        input_quality=_payload_input_quality(payload),
    )

