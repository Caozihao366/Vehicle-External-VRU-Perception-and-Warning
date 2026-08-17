"""Track-ID aligned rolling history for real-time BEV input."""

from collections import deque
import math
from typing import Callable, Deque, Dict, List, Optional, Protocol, Sequence, Tuple

import torch

from .bev_types import BEVFrame, EgoState, ObjectState


AGENT_TYPE_TO_INDEX = {
    'vehicle': 0,
    'pedestrian': 1,
    'motorcyclist': 2,
    'cyclist': 3,
    'bus': 4,
    'unknown': 9,
}


class CoordinateWindowAdapter(Protocol):
    """Optional hook that re-expresses a complete window in its latest t0 frame."""

    def __call__(self, frames: Sequence[BEVFrame]) -> Sequence[BEVFrame]:
        ...


def identity_coordinate_adapter(frames: Sequence[BEVFrame]) -> Sequence[BEVFrame]:
    """Do not transform coordinates; all frames must already share one Ego frame."""

    return frames


class BEVHistoryBuffer:
    """Keep at most 50 BEV frames and construct the existing predictor input."""

    def __init__(
        self,
        history_steps: int = 50,
        frequency_hz: float = 10.0,
        timestamp_tolerance_seconds: float = 0.03,
        coordinate_adapter: Optional[CoordinateWindowAdapter] = None,
    ) -> None:
        if history_steps != 50:
            raise ValueError('The deployed checkpoint requires history_steps=50.')
        if abs(float(frequency_hz) - 10.0) > 1e-6:
            raise ValueError('The deployed checkpoint requires frequency_hz=10.0.')
        if timestamp_tolerance_seconds < 0:
            raise ValueError('timestamp_tolerance_seconds must be non-negative.')
        self.history_steps = int(history_steps)
        self.frequency_hz = float(frequency_hz)
        self.timestamp_tolerance_seconds = float(timestamp_tolerance_seconds)
        self.coordinate_adapter = coordinate_adapter or identity_coordinate_adapter
        self._frames: Deque[BEVFrame] = deque(maxlen=self.history_steps)
        self._ego_id: Optional[str] = None

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    @property
    def is_full(self) -> bool:
        return self.num_frames == self.history_steps

    @property
    def track_ids(self) -> List[str]:
        """Track IDs still represented anywhere in the rolling window."""

        return self._ordered_track_ids(list(self._frames)) if self._frames else []

    def clear(self) -> None:
        self._frames.clear()
        self._ego_id = None

    @staticmethod
    def _validate_vector(value: object, name: str) -> None:
        try:
            tensor = torch.as_tensor(value, dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name} must contain two numeric values.') from error
        if tuple(tensor.shape) != (2,):
            raise ValueError(f'{name} must have shape [2], got {tuple(tensor.shape)}.')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'{name} contains NaN or Inf.')

    @classmethod
    def _validate_state(cls, state: object, name: str) -> None:
        if not isinstance(state, (EgoState, ObjectState)):
            raise ValueError(f'{name} has an invalid state type.')
        if state.id is None or str(state.id) == '':
            raise ValueError(f'{name}.id must not be empty.')
        cls._validate_vector(state.position, f'{name}.position')
        cls._validate_vector(state.velocity, f'{name}.velocity')
        try:
            heading = float(state.heading)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name}.heading must be numeric.') from error
        if not math.isfinite(heading):
            raise ValueError(f'{name}.heading contains NaN or Inf.')

    def update(self, frame: BEVFrame) -> None:
        if not isinstance(frame, BEVFrame):
            raise ValueError('frame must be a BEVFrame instance.')
        if frame.coordinate_frame != 'ego':
            raise ValueError("BEVFrame.coordinate_frame must be exactly 'ego'.")
        try:
            timestamp = float(frame.timestamp)
        except (TypeError, ValueError) as error:
            raise ValueError('BEVFrame.timestamp must be numeric seconds.') from error
        if not math.isfinite(timestamp):
            raise ValueError('BEVFrame.timestamp contains NaN or Inf.')
        if self._frames:
            delta = timestamp - float(self._frames[-1].timestamp)
            expected = 1.0 / self.frequency_hz
            if delta <= 0:
                raise ValueError('BEVFrame timestamps must be strictly increasing.')
            if abs(delta - expected) > self.timestamp_tolerance_seconds:
                raise ValueError(
                    f'Expected approximately 10 Hz (delta {expected:.3f}s), '
                    f'got {delta:.6f}s. Resample upstream or reset the stream.')

        self._validate_state(frame.ego, 'ego')
        ego_id = str(frame.ego.id)
        if self._ego_id is not None and ego_id != self._ego_id:
            raise ValueError('Ego ID changed; clear the buffer before starting a new stream.')
        seen_ids = {ego_id}
        for index, obj in enumerate(frame.objects):
            self._validate_state(obj, f'objects[{index}]')
            object_id = str(obj.id)
            if object_id in seen_ids:
                raise ValueError(f'Duplicate Track ID in one BEVFrame: {object_id!r}.')
            seen_ids.add(object_id)

        self._frames.append(frame)
        self._ego_id = ego_id

    @staticmethod
    def _state_map(frame: BEVFrame) -> Dict[str, object]:
        states: Dict[str, object] = {str(frame.ego.id): frame.ego}
        states.update({str(obj.id): obj for obj in frame.objects})
        return states

    @classmethod
    def _ordered_track_ids(cls, frames: Sequence[BEVFrame]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        ego_id = str(frames[-1].ego.id)
        ordered.append(ego_id)
        seen.add(ego_id)
        for frame in frames:
            for obj in frame.objects:
                track_id = str(obj.id)
                if track_id not in seen:
                    ordered.append(track_id)
                    seen.add(track_id)
        return ordered

    @staticmethod
    def _agent_type(state: object) -> int:
        if isinstance(state, EgoState):
            return AGENT_TYPE_TO_INDEX['vehicle']
        name = str(state.type).strip().lower()
        return AGENT_TYPE_TO_INDEX.get(name, AGENT_TYPE_TO_INDEX['unknown'])

    def build_model_input(self, minimum_valid_samples: int = 1) -> Dict[str, object]:
        """Return a left-padded dictionary accepted by EgoTrajectoryPredictor.predict().

        ``minimum_valid_samples=2`` is used by the real-time wrapper so a newly
        appeared track cannot violate the model's minimum motion-history rule.
        The track remains in this buffer and is admitted after its second sample.
        """

        if not self._frames:
            raise RuntimeError('Cannot build model input from an empty history buffer.')
        if minimum_valid_samples < 1:
            raise ValueError('minimum_valid_samples must be at least 1.')

        raw_frames = list(self._frames)
        frames = list(self.coordinate_adapter(tuple(raw_frames)))
        if len(frames) != len(raw_frames):
            raise ValueError('coordinate_adapter must return exactly one frame per input frame.')
        if not frames:
            raise ValueError('coordinate_adapter returned an empty window.')
        if any(frame.coordinate_frame != 'ego' for frame in frames):
            raise ValueError('coordinate_adapter output must remain in coordinate_frame="ego".')

        state_maps = [self._state_map(frame) for frame in frames]
        ordered_ids = self._ordered_track_ids(frames)
        counts = {
            track_id: sum(track_id in state_map for state_map in state_maps)
            for track_id in ordered_ids
        }
        ego_id = str(frames[-1].ego.id)
        if counts.get(ego_id, 0) < minimum_valid_samples:
            raise RuntimeError(
                f'AV needs at least {minimum_valid_samples} valid historical samples.')
        agent_ids = [
            track_id for track_id in ordered_ids
            if counts[track_id] >= minimum_valid_samples
        ]

        num_agents = len(agent_ids)
        position = torch.zeros(num_agents, self.history_steps, 2, dtype=torch.float32)
        velocity = torch.zeros_like(position)
        heading = torch.zeros(num_agents, self.history_steps, dtype=torch.float32)
        valid_mask = torch.zeros(num_agents, self.history_steps, dtype=torch.bool)
        agent_type = torch.full(
            (num_agents,), AGENT_TYPE_TO_INDEX['unknown'], dtype=torch.long)
        left_offset = self.history_steps - len(frames)

        latest_state_by_id: Dict[str, object] = {}
        for state_map in state_maps:
            latest_state_by_id.update(state_map)
        for agent_index, track_id in enumerate(agent_ids):
            agent_type[agent_index] = self._agent_type(latest_state_by_id[track_id])
            for local_index, state_map in enumerate(state_maps):
                state = state_map.get(track_id)
                if state is None:
                    continue
                history_index = left_offset + local_index
                position[agent_index, history_index] = torch.as_tensor(
                    state.position, dtype=torch.float32)
                velocity[agent_index, history_index] = torch.as_tensor(
                    state.velocity, dtype=torch.float32)
                heading_value = float(state.heading)
                heading[agent_index, history_index] = math.atan2(
                    math.sin(heading_value), math.cos(heading_value))
                valid_mask[agent_index, history_index] = True

        return {
            'position': position,
            'velocity': velocity,
            'heading': heading,
            'valid_mask': valid_mask,
            'agent_type': agent_type,
            'av_index': agent_ids.index(ego_id),
            'agent_ids': agent_ids,
            'coordinate_frame': 'ego',
            'frequency_hz': self.frequency_hz,
        }
