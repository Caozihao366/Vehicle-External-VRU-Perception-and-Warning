"""Public API for the trajectory-based TTC risk package."""

from .adapter import compute_multimodal_ttc_from_payload
from .core import (
    build_ego_trajectory,
    circle_to_oriented_box_clearance,
    compute_multimodal_ttc,
    compute_path_ttc,
    compute_ttc,
    make_cv_prediction_mode,
    predict_ego,
    predict_vru,
)
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
from .risk import RiskStateMachine, classify_risk

__all__ = [
    "EgoState",
    "VRUState",
    "TTCResult",
    "EgoTrajectoryPoint",
    "VRUTrajectoryPoint",
    "PredictionMode",
    "ModeTTCResult",
    "MultimodalTTCResult",
    "RiskStateMachine",
    "predict_ego",
    "predict_vru",
    "circle_to_oriented_box_clearance",
    "compute_ttc",
    "build_ego_trajectory",
    "make_cv_prediction_mode",
    "compute_path_ttc",
    "classify_risk",
    "compute_multimodal_ttc",
    "compute_multimodal_ttc_from_payload",
]
