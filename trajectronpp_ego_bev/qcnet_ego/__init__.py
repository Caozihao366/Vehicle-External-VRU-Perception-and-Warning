"""Deployment-only API for Ego-coordinate, map-free QCNet."""

from .output_types import PredictionResult
from .predictor import EgoTrajectoryPredictor

__all__ = ['EgoTrajectoryPredictor', 'PredictionResult']
