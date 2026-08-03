from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass(frozen=True)
class PredictionResult:
    trajectories: torch.Tensor
    probabilities: torch.Tensor
    agent_indices: torch.Tensor
    agent_ids: Optional[List[str]]
    coordinate_frame: str = 'ego'
    reference_time: str = 'history_last_t0'
    horizon_steps: int = 20
    horizon_seconds: float = 2.0
    frequency_hz: float = 10.0
    num_modes: int = 6
    position_unit: str = 'meter'

    def to_dict(self) -> Dict[str, object]:
        return {
            'trajectories': self.trajectories,
            'probabilities': self.probabilities,
            'agent_indices': self.agent_indices,
            'agent_ids': self.agent_ids,
            'coordinate_frame': self.coordinate_frame,
            'reference_time': self.reference_time,
            'horizon_steps': self.horizon_steps,
            'horizon_seconds': self.horizon_seconds,
            'frequency_hz': self.frequency_hz,
            'num_modes': self.num_modes,
            'position_unit': self.position_unit,
        }

    def save(self, path: str) -> None:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), output_path)
