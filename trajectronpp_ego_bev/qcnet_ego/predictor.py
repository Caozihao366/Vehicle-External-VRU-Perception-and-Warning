"""Stable high-level single-scenario deployment interface."""

from typing import Mapping

import torch

from .input_adapter import EgoInputAdapter
from .model_loader import load_inference_model
from .output_types import PredictionResult


class EgoTrajectoryPredictor:
    """Run map-free, Ego-coordinate QCNet inference for one scenario at a time."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        return_cpu: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.return_cpu = bool(return_cpu)
        self.input_adapter = EgoInputAdapter()
        self.model, self.load_audit = load_inference_model(checkpoint_path, device)

    def predict(self, input_data: Mapping[str, object]) -> PredictionResult:
        if self.model is None:
            raise RuntimeError('Predictor is closed.')
        adapted = self.input_adapter.adapt(input_data)
        prediction_indices = adapted.prediction_indices.to(self.device)
        graph = adapted.data.to(self.device)

        self.model.eval()
        with torch.inference_mode():
            raw_prediction = self.model(graph)
            trajectories = raw_prediction['loc_refine_pos'][
                prediction_indices,
                :,
                :self.model.prediction_horizon_steps,
                :self.model.output_dim,
            ]
            probabilities = torch.softmax(
                raw_prediction['pi'][prediction_indices], dim=-1)

        expected_trajectory_shape = (
            prediction_indices.numel(), self.model.num_modes,
            self.model.prediction_horizon_steps, self.model.output_dim,
        )
        if tuple(trajectories.shape) != expected_trajectory_shape:
            raise RuntimeError(
                f'Unexpected trajectory output shape: {tuple(trajectories.shape)}, '
                f'expected {expected_trajectory_shape}.')
        if tuple(probabilities.shape) != (
            prediction_indices.numel(), self.model.num_modes,
        ):
            raise RuntimeError(f'Unexpected probability shape: {tuple(probabilities.shape)}.')
        if not torch.isfinite(trajectories).all():
            raise RuntimeError('Model trajectories contain NaN or Inf.')
        if not torch.isfinite(probabilities).all():
            raise RuntimeError('Model probabilities contain NaN or Inf.')
        probability_sums = probabilities.sum(dim=-1)
        if not torch.allclose(
            probability_sums, torch.ones_like(probability_sums), atol=1e-5, rtol=1e-5,
        ):
            raise RuntimeError('Mode probabilities do not sum to one.')

        if self.return_cpu:
            trajectories = trajectories.cpu()
            probabilities = probabilities.cpu()
            result_indices = adapted.prediction_indices.cpu()
        else:
            result_indices = prediction_indices
        return PredictionResult(
            trajectories=trajectories,
            probabilities=probabilities,
            agent_indices=result_indices,
            agent_ids=adapted.prediction_agent_ids,
            horizon_steps=self.model.prediction_horizon_steps,
            horizon_seconds=(
                self.model.prediction_horizon_steps
                / self.model.trajectory_frequency_hz
            ),
            frequency_hz=self.model.trajectory_frequency_hz,
            num_modes=self.model.num_modes,
        )

    def close(self) -> None:
        self.model = None
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
