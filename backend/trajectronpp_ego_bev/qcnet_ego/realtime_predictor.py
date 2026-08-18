"""Composition layer connecting BEV frames to the existing inference predictor."""

from typing import Optional

from .bev_types import BEVFrame
from .history_buffer import BEVHistoryBuffer, CoordinateWindowAdapter
from .output_types import PredictionResult
from .predictor import EgoTrajectoryPredictor


class RealtimeTrajectoryPredictor:
    """Stateful single-stream BEV-to-QCNet deployment interface."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        return_cpu: bool = True,
        coordinate_adapter: Optional[CoordinateWindowAdapter] = None,
    ) -> None:
        self.history_buffer = BEVHistoryBuffer(
            coordinate_adapter=coordinate_adapter,
        )
        self.predictor = EgoTrajectoryPredictor(
            checkpoint_path=checkpoint_path,
            device=device,
            return_cpu=return_cpu,
        )

    def update(self, bev_frame: BEVFrame) -> Optional[PredictionResult]:
        """Consume one frame; predict after at least two valid AV samples."""

        self.history_buffer.update(bev_frame)
        if self.history_buffer.num_frames < 2:
            return None
        model_input = self.history_buffer.build_model_input(minimum_valid_samples=2)
        return self.predictor.predict(model_input)

    def reset_history(self) -> None:
        self.history_buffer.clear()

    def close(self) -> None:
        self.history_buffer.clear()
        self.predictor.close()
