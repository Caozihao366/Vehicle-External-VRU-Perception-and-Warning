"""Direct in-process QCNet trajectory provider for TTC.

This integration is imported only when QCNet inference is required. The TTC
core remains independent of torch and torch-geometric.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ..models import PredictionMode, VRUTrajectoryPoint


DEFAULT_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "trajectronpp_ego_bev"
)


@dataclass(frozen=True)
class QCNetTargetPrediction:
    """TTC-ready multimodal prediction for one tracked target."""

    track_id: str
    modes: tuple[PredictionMode, ...]
    inference_time_ms: float
    prediction_source: str = "qcnet_ego_2s_6mode"
    probability_semantics: str = (
        "model_softmax_relative_mode_probability"
    )
    prediction_horizon_s: float = 2.0


class QCNetTrajectoryProvider:
    """Load QCNet once and predict selected targets in the current scenario."""

    def __init__(
        self,
        package_root: Path | str = DEFAULT_PACKAGE_ROOT,
        *,
        device: str = "cuda",
    ) -> None:
        self.package_root = Path(package_root).resolve()
        checkpoint = (
            self.package_root
            / "checkpoints"
            / "QCNet_Ego_2s_best.ckpt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        package_path = str(self.package_root)
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
        from qcnet_ego import EgoTrajectoryPredictor

        self.device = device
        self._predictor = EgoTrajectoryPredictor(
            str(checkpoint),
            device=device,
            return_cpu=True,
        )

    def predict(
        self,
        input_data: Mapping[str, object],
        *,
        target_ids: Sequence[str],
    ) -> dict[str, QCNetTargetPrediction]:
        """Return TTC-ready modes for every requested track id."""

        ids = input_data.get("agent_ids")
        if not isinstance(ids, (list, tuple)):
            raise ValueError(
                "agent_ids are required for stable TTC target matching"
            )
        requested = tuple(str(track_id) for track_id in target_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("target_ids must be unique")

        positions = np.asarray(input_data["position"], dtype=float)
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError("position must have shape [N, H, 2]")
        current_positions = positions[:, -1, :]

        started = time.perf_counter()
        result = self._predictor.predict(input_data)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0

        output_ids = result.agent_ids or [
            str(ids[index]) for index in result.agent_indices.tolist()
        ]
        if len(set(output_ids)) != len(output_ids):
            raise RuntimeError("QCNet returned duplicate agent_ids")

        output: dict[str, QCNetTargetPrediction] = {}
        requested_set = set(requested)
        for output_index, track_id in enumerate(output_ids):
            track_id = str(track_id)
            if track_id not in requested_set:
                continue

            input_index = int(result.agent_indices[output_index])
            probabilities = result.probabilities[output_index].numpy()
            trajectories = result.trajectories[output_index].numpy()
            modes = []
            for mode_index, (probability, trajectory) in enumerate(
                zip(probabilities, trajectories)
            ):
                probability = float(probability)
                if not math.isfinite(probability) or probability < 0:
                    raise RuntimeError(
                        "QCNet returned an invalid mode probability"
                    )

                # QCNet outputs t0+0.1 ... t0+2.0. TTC requires t=0,
                # therefore prepend the current tracked metric position.
                points = [
                    VRUTrajectoryPoint(
                        t_s=0.0,
                        x=float(current_positions[input_index, 0]),
                        y=float(current_positions[input_index, 1]),
                    )
                ]
                points.extend(
                    VRUTrajectoryPoint(
                        t_s=(step + 1) / float(result.frequency_hz),
                        x=float(xy[0]),
                        y=float(xy[1]),
                    )
                    for step, xy in enumerate(trajectory)
                )
                modes.append(
                    PredictionMode(
                        mode_id=f"qcnet_mode_{mode_index}",
                        probability=probability,
                        points=tuple(points),
                    )
                )

            output[track_id] = QCNetTargetPrediction(
                track_id=track_id,
                modes=tuple(modes),
                inference_time_ms=inference_ms,
                prediction_horizon_s=float(result.horizon_seconds),
            )

        missing = requested_set - set(output)
        if missing:
            raise RuntimeError(
                f"QCNet returned no prediction for targets: {sorted(missing)}"
            )
        return output

    def close(self) -> None:
        self._predictor.close()

    def __enter__(self) -> "QCNetTrajectoryProvider":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
