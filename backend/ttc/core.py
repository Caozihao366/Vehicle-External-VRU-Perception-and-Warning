"""Trajectory prediction, geometry, collision search, and multimodal TTC."""

from __future__ import annotations

import bisect
import math
from typing import Sequence

from .models import (
    EgoState,
    EgoTrajectoryPoint,
    ModeTTCResult,
    MultimodalTTCResult,
    PredictionMode,
    TTCResult,
    VRUState,
    VRUTrajectoryPoint,
)
from .risk import classify_risk


def predict_ego(state: EgoState, time_s: float) -> EgoState:
    """Predict an ego state using constant speed and constant yaw rate."""
    if abs(state.yaw_rate_rps) < 1e-8:
        x = state.x + state.speed_mps * math.cos(state.yaw) * time_s
        y = state.y + state.speed_mps * math.sin(state.yaw) * time_s
    else:
        next_yaw = state.yaw + state.yaw_rate_rps * time_s
        radius = state.speed_mps / state.yaw_rate_rps
        x = state.x + radius * (math.sin(next_yaw) - math.sin(state.yaw))
        y = state.y + radius * (-math.cos(next_yaw) + math.cos(state.yaw))

    return EgoState(
        x=x,
        y=y,
        yaw=state.yaw + state.yaw_rate_rps * time_s,
        speed_mps=state.speed_mps,
        yaw_rate_rps=state.yaw_rate_rps,
    )


def predict_vru(state: VRUState, time_s: float) -> VRUState:
    """Predict a VRU state using constant Cartesian velocity."""
    return VRUState(
        x=state.x + state.vx_mps * time_s,
        y=state.y + state.vy_mps * time_s,
        vx_mps=state.vx_mps,
        vy_mps=state.vy_mps,
        radius_m=state.radius_m,
    )


def circle_to_oriented_box_clearance(
    circle_x: float,
    circle_y: float,
    circle_radius: float,
    box_x: float,
    box_y: float,
    box_yaw: float,
    box_length: float,
    box_width: float,
) -> float:
    """Return signed clearance between a circle and an oriented rectangle.

    A value <= 0 means the two footprints intersect.
    """
    dx = circle_x - box_x
    dy = circle_y - box_y
    cos_yaw = math.cos(box_yaw)
    sin_yaw = math.sin(box_yaw)

    # Rotate the circle centre into the rectangle's local coordinates.
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    outside_x = max(abs(local_x) - box_length / 2.0, 0.0)
    outside_y = max(abs(local_y) - box_width / 2.0, 0.0)
    return math.hypot(outside_x, outside_y) - circle_radius


def compute_ttc(
    ego: EgoState,
    vru: VRUState,
    *,
    horizon_s: float = 5.0,
    dt_s: float = 0.1,
    ego_length_m: float = 4.8,
    ego_width_m: float = 2.0,
    safety_margin_m: float = 0.4,
) -> TTCResult:
    """Compute the first sampled collision time over the prediction horizon."""
    if horizon_s < 0:
        raise ValueError("horizon_s must be non-negative")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if min(ego_length_m, ego_width_m) <= 0:
        raise ValueError("ego dimensions must be positive")
    if safety_margin_m < 0 or vru.radius_m < 0:
        raise ValueError("footprint margins must be non-negative")

    inflated_length = ego_length_m + 2.0 * safety_margin_m
    inflated_width = ego_width_m + 2.0 * safety_margin_m
    steps = int(math.floor(horizon_s / dt_s + 1e-9))

    best_clearance = math.inf
    best_time = 0.0
    collision_time = None
    collision_x = None
    collision_y = None

    for step in range(steps + 1):
        time_s = step * dt_s
        ego_at_t = predict_ego(ego, time_s)
        vru_at_t = predict_vru(vru, time_s)
        clearance = circle_to_oriented_box_clearance(
            vru_at_t.x,
            vru_at_t.y,
            vru_at_t.radius_m,
            ego_at_t.x,
            ego_at_t.y,
            ego_at_t.yaw,
            inflated_length,
            inflated_width,
        )

        if clearance < best_clearance:
            best_clearance = clearance
            best_time = time_s

        if clearance <= 0.0:
            collision_time = time_s
            collision_x = vru_at_t.x
            collision_y = vru_at_t.y
            break

    return TTCResult(
        ttc_s=collision_time,
        collision=collision_time is not None,
        min_clearance_m=best_clearance,
        min_clearance_time_s=best_time,
        collision_x_m=collision_x,
        collision_y_m=collision_y,
    )


def build_ego_trajectory(
    state: EgoState,
    *,
    horizon_s: float,
    dt_s: float,
) -> tuple[EgoTrajectoryPoint, ...]:
    """Sample a CTRV ego trajectory for consumption by the generic TTC API."""
    if horizon_s < 0:
        raise ValueError("horizon_s must be non-negative")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    steps = int(math.floor(horizon_s / dt_s + 1e-9))
    points = []
    for step in range(steps + 1):
        time_s = step * dt_s
        predicted = predict_ego(state, time_s)
        points.append(
            EgoTrajectoryPoint(
                t_s=time_s,
                x=predicted.x,
                y=predicted.y,
                yaw=predicted.yaw,
            )
        )
    return tuple(points)


def make_cv_prediction_mode(
    state: VRUState,
    *,
    horizon_s: float,
    dt_s: float,
    mode_id: str = "cv_fallback",
    probability: float = 1.0,
) -> PredictionMode:
    """Build a constant-velocity fallback mode when no learned model is ready."""
    if horizon_s < 0:
        raise ValueError("horizon_s must be non-negative")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    steps = int(math.floor(horizon_s / dt_s + 1e-9))
    points = []
    for step in range(steps + 1):
        time_s = step * dt_s
        predicted = predict_vru(state, time_s)
        points.append(VRUTrajectoryPoint(time_s, predicted.x, predicted.y))
    return PredictionMode(mode_id=mode_id, probability=probability, points=tuple(points))


def _validate_time_series(points: Sequence[object], label: str) -> None:
    if not points:
        raise ValueError(f"{label} must contain at least one point")
    times = [float(getattr(point, "t_s")) for point in points]
    if abs(times[0]) > 1e-9:
        raise ValueError(f"{label} must start at t_s=0")
    if any(not math.isfinite(time_s) or time_s < 0 for time_s in times):
        raise ValueError(f"{label} contains an invalid timestamp")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(f"{label} timestamps must be strictly increasing")
    for point in points:
        numeric_fields = (
            value
            for name, value in vars(point).items()
            if name != "t_s" and isinstance(value, (int, float))
        )
        if any(not math.isfinite(float(value)) for value in numeric_fields):
            raise ValueError(f"{label} contains a non-finite value")


def _interpolate_ego(
    points: Sequence[EgoTrajectoryPoint],
    time_s: float,
    times: Sequence[float] | None = None,
) -> EgoTrajectoryPoint:
    if time_s <= points[0].t_s:
        return points[0]
    if time_s >= points[-1].t_s:
        return points[-1]

    time_values = times if times is not None else tuple(point.t_s for point in points)
    right_index = bisect.bisect_right(time_values, time_s)
    left, right = points[right_index - 1], points[right_index]
    alpha = (time_s - left.t_s) / (right.t_s - left.t_s)
    yaw_delta = (right.yaw - left.yaw + math.pi) % (2.0 * math.pi) - math.pi
    return EgoTrajectoryPoint(
        t_s=time_s,
        x=left.x + alpha * (right.x - left.x),
        y=left.y + alpha * (right.y - left.y),
        yaw=left.yaw + alpha * yaw_delta,
    )


def _interpolate_vru(
    points: Sequence[VRUTrajectoryPoint],
    time_s: float,
    times: Sequence[float] | None = None,
) -> VRUTrajectoryPoint:
    if time_s <= points[0].t_s:
        return points[0]
    if time_s >= points[-1].t_s:
        return points[-1]

    time_values = times if times is not None else tuple(point.t_s for point in points)
    right_index = bisect.bisect_right(time_values, time_s)
    left, right = points[right_index - 1], points[right_index]
    alpha = (time_s - left.t_s) / (right.t_s - left.t_s)
    return VRUTrajectoryPoint(
        t_s=time_s,
        x=left.x + alpha * (right.x - left.x),
        y=left.y + alpha * (right.y - left.y),
        sigma_x_m=left.sigma_x_m
        + alpha * (right.sigma_x_m - left.sigma_x_m),
        sigma_y_m=left.sigma_y_m
        + alpha * (right.sigma_y_m - left.sigma_y_m),
    )


def compute_path_ttc(
    ego_trajectory: Sequence[EgoTrajectoryPoint],
    vru_trajectory: Sequence[VRUTrajectoryPoint],
    *,
    dt_s: float = 0.1,
    ego_length_m: float = 4.8,
    ego_width_m: float = 2.0,
    safety_margin_m: float = 0.4,
    vru_radius_m: float = 0.4,
    uncertainty_base_m: float = 0.15,
    uncertainty_growth_mps: float = 0.20,
    sigma_scale: float = 1.0,
    collision_check_dt_s: float = 0.02,
) -> TTCResult:
    """Compute TTC for two externally supplied, time-aligned trajectories.

    Uncertainty is conservatively approximated by expanding the VRU radius:
    radius(t) = physical radius + base + growth*t + sigma_scale*max(sigma_x,sigma_y).
    This is an occupancy approximation, not a Gaussian collision probability.
    """
    _validate_time_series(ego_trajectory, "ego_trajectory")
    _validate_time_series(vru_trajectory, "vru_trajectory")
    if dt_s <= 0 or collision_check_dt_s <= 0:
        raise ValueError("time steps must be positive")
    if min(ego_length_m, ego_width_m) <= 0:
        raise ValueError("ego dimensions must be positive")
    if min(
        safety_margin_m,
        vru_radius_m,
        uncertainty_base_m,
        uncertainty_growth_mps,
        sigma_scale,
    ) < 0:
        raise ValueError("footprint and uncertainty parameters must be non-negative")

    horizon_s = min(ego_trajectory[-1].t_s, vru_trajectory[-1].t_s)
    # Input paths may be sparse (for example 0.5 or 1.0 s apart).  Collision
    # checking uses its own finer interval so a VRU cannot pass through the
    # ego envelope between two ordinary 0.1 s samples.
    check_dt_s = min(dt_s, collision_check_dt_s)
    steps = int(math.ceil(horizon_s / check_dt_s - 1e-12))
    inflated_length = ego_length_m + 2.0 * safety_margin_m
    inflated_width = ego_width_m + 2.0 * safety_margin_m
    best_clearance = math.inf
    best_time = 0.0
    collision_time = None
    collision_x = None
    collision_y = None
    ego_times = tuple(point.t_s for point in ego_trajectory)
    vru_times = tuple(point.t_s for point in vru_trajectory)

    for step in range(steps + 1):
        time_s = min(step * check_dt_s, horizon_s)
        ego_at_t = _interpolate_ego(ego_trajectory, time_s, ego_times)
        vru_at_t = _interpolate_vru(vru_trajectory, time_s, vru_times)
        effective_radius = (
            vru_radius_m
            + uncertainty_base_m
            + uncertainty_growth_mps * time_s
            + sigma_scale * max(vru_at_t.sigma_x_m, vru_at_t.sigma_y_m)
        )
        clearance = circle_to_oriented_box_clearance(
            vru_at_t.x,
            vru_at_t.y,
            effective_radius,
            ego_at_t.x,
            ego_at_t.y,
            ego_at_t.yaw,
            inflated_length,
            inflated_width,
        )
        if clearance < best_clearance:
            best_clearance = clearance
            best_time = time_s
        if clearance <= 0.0 and collision_time is None:
            collision_time = time_s
            collision_x = vru_at_t.x
            collision_y = vru_at_t.y

    return TTCResult(
        ttc_s=collision_time,
        collision=collision_time is not None,
        min_clearance_m=best_clearance,
        min_clearance_time_s=best_time,
        collision_x_m=collision_x,
        collision_y_m=collision_y,
    )


def _seconds_to_ms(value_s: float | None) -> int | None:
    return None if value_s is None else int(round(value_s * 1000.0))


def compute_multimodal_ttc(
    track_id: int | str,
    ego_trajectory: Sequence[EgoTrajectoryPoint],
    modes: Sequence[PredictionMode],
    *,
    prediction_source: str = "intent_model",
    behavior_state: str = "normal",
    sensor_availability: str = "available",
    dt_s: float = 0.1,
    ego_length_m: float = 4.8,
    ego_width_m: float = 2.0,
    safety_margin_m: float = 0.4,
    vru_radius_m: float = 0.4,
    uncertainty_base_m: float = 0.15,
    uncertainty_growth_mps: float = 0.20,
    sigma_scale: float = 1.0,
    danger_ttc_s: float = 1.2,
    warning_ttc_s: float = 2.5,
    collision_check_dt_s: float = 0.02,
    ego_trajectory_source: str = "external",
    input_quality: str = "valid",
) -> MultimodalTTCResult:
    """Evaluate every prediction mode independently and aggregate modal mass."""
    _validate_time_series(ego_trajectory, "ego_trajectory")
    if not modes:
        raise ValueError("at least one prediction mode is required")
    if any(not math.isfinite(mode.probability) or mode.probability < 0 for mode in modes):
        raise ValueError("mode probabilities must be finite and non-negative")
    total_probability = sum(mode.probability for mode in modes)
    if total_probability <= 0:
        raise ValueError("sum of mode probabilities must be positive")
    if len({mode.mode_id for mode in modes}) != len(modes):
        raise ValueError("mode_id values must be unique")
    if sensor_availability not in {"available", "degraded", "unavailable"}:
        raise ValueError(
            "sensor_availability must be available, degraded, or unavailable"
        )

    evaluated_horizon_s = min(
        ego_trajectory[-1].t_s,
        max(mode.points[-1].t_s if mode.points else 0.0 for mode in modes),
    )
    results = []
    for mode in modes:
        normalized_probability = mode.probability / total_probability
        result = compute_path_ttc(
            ego_trajectory,
            mode.points,
            dt_s=dt_s,
            ego_length_m=ego_length_m,
            ego_width_m=ego_width_m,
            safety_margin_m=safety_margin_m,
            vru_radius_m=vru_radius_m,
            uncertainty_base_m=uncertainty_base_m,
            uncertainty_growth_mps=uncertainty_growth_mps,
            sigma_scale=sigma_scale,
            collision_check_dt_s=collision_check_dt_s,
        )
        results.append(
            ModeTTCResult(
                mode_id=mode.mode_id,
                probability=normalized_probability,
                ttc_ms=_seconds_to_ms(result.ttc_s),
                collision=result.collision,
                min_clearance_m=result.min_clearance_m,
                min_clearance_time_s=result.min_clearance_time_s,
                collision_x_m=result.collision_x_m,
                collision_y_m=result.collision_y_m,
            )
        )

    colliding = [result for result in results if result.collision]
    earliest_ttc_ms = min(
        (result.ttc_ms for result in colliding if result.ttc_ms is not None),
        default=None,
    )
    earliest_ttc = (
        None if earliest_ttc_ms is None else earliest_ttc_ms / 1000.0
    )
    collision_probability = sum(result.probability for result in colliding)
    most_likely = max(results, key=lambda result: result.probability)
    critical = min(
        colliding,
        key=lambda result: (
            result.ttc_ms if result.ttc_ms is not None else math.inf,
            -result.probability,
        ),
        default=None,
    )
    closest = min(results, key=lambda result: result.min_clearance_m)
    risk_level, major_trigger_reason = classify_risk(
        earliest_ttc,
        collision_probability,
        behavior_state=behavior_state,
        danger_ttc_s=danger_ttc_s,
        warning_ttc_s=warning_ttc_s,
    )
    if input_quality != "valid":
        status = input_quality.upper()
        risk_level = "notice"
        major_trigger_reason = input_quality
    elif sensor_availability == "unavailable":
        status = "SENSOR_UNAVAILABLE"
        risk_level = "notice"
        major_trigger_reason = "sensor_unavailable"
    elif colliding:
        status = "COLLISION_PREDICTED"
    else:
        status = "NO_COLLISION_WITHIN_HORIZON"

    return MultimodalTTCResult(
        track_id=track_id,
        prediction_source=prediction_source,
        sensor_availability=sensor_availability,
        behavior_state=behavior_state,
        evaluated_horizon_ms=_seconds_to_ms(evaluated_horizon_s) or 0,
        earliest_ttc_ms=earliest_ttc_ms,
        most_likely_mode_id=most_likely.mode_id,
        most_likely_ttc_ms=most_likely.ttc_ms,
        collision_probability=collision_probability,
        risk_level=risk_level,
        major_trigger_reason=major_trigger_reason,
        requires_aeb=risk_level == "danger" and sensor_availability != "unavailable",
        status=status,
        ego_trajectory_source=ego_trajectory_source,
        input_quality=input_quality,
        ttc_resolution_ms=_seconds_to_ms(min(dt_s, collision_check_dt_s)) or 0,
        critical_mode_id=None if critical is None else critical.mode_id,
        critical_mode_probability=0.0 if critical is None else critical.probability,
        collision_probability_semantics="colliding_mode_probability_mass",
        min_clearance_m=closest.min_clearance_m,
        time_to_min_clearance_ms=_seconds_to_ms(closest.min_clearance_time_s) or 0,
        mode_results=tuple(results),
    )

