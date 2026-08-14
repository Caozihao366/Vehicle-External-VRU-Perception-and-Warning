"""Public data contracts for the trajectory-based TTC baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EgoState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed_mps: float = 0.0
    yaw_rate_rps: float = 0.0


@dataclass(frozen=True)
class VRUState:
    x: float
    y: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    radius_m: float = 0.4


@dataclass(frozen=True)
class TTCResult:
    ttc_s: float | None
    collision: bool
    min_clearance_m: float
    min_clearance_time_s: float
    collision_x_m: float | None
    collision_y_m: float | None


@dataclass(frozen=True)
class EgoTrajectoryPoint:
    t_s: float
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class VRUTrajectoryPoint:
    t_s: float
    x: float
    y: float
    sigma_x_m: float = 0.0
    sigma_y_m: float = 0.0


@dataclass(frozen=True)
class PredictionMode:
    mode_id: str
    probability: float
    points: tuple[VRUTrajectoryPoint, ...]


@dataclass(frozen=True)
class ModeTTCResult:
    mode_id: str
    probability: float
    ttc_ms: int | None
    collision: bool
    min_clearance_m: float
    min_clearance_time_s: float
    collision_x_m: float | None
    collision_y_m: float | None


@dataclass(frozen=True)
class MultimodalTTCResult:
    track_id: int | str
    prediction_source: str
    sensor_availability: str
    behavior_state: str
    evaluated_horizon_ms: int
    earliest_ttc_ms: int | None
    most_likely_mode_id: str
    most_likely_ttc_ms: int | None
    collision_probability: float
    risk_level: str
    major_trigger_reason: str
    requires_aeb: bool
    status: str
    ego_trajectory_source: str
    input_quality: str
    ttc_resolution_ms: int
    critical_mode_id: str | None
    critical_mode_probability: float
    collision_probability_semantics: str
    min_clearance_m: float
    time_to_min_clearance_ms: int
    mode_results: tuple[ModeTTCResult, ...]
