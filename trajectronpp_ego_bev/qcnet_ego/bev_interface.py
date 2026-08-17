"""Backward-compatible imports for the split real-time BEV interface."""

from .bev_types import BEVFrame, EgoState, ObjectState
from .history_buffer import BEVHistoryBuffer, CoordinateWindowAdapter
from .realtime_predictor import RealtimeTrajectoryPredictor

HistoryBuffer = BEVHistoryBuffer

__all__ = [
    'BEVFrame',
    'BEVHistoryBuffer',
    'CoordinateWindowAdapter',
    'EgoState',
    'HistoryBuffer',
    'ObjectState',
    'RealtimeTrajectoryPredictor',
]
