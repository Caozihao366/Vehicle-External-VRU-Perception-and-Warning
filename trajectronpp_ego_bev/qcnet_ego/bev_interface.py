"""Real-time Python-object interface from BEV tracks to one QCNet scenario."""

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .input_adapter import AGENT_TYPES


AgentType = Union[str, int]
Vector2 = Sequence[float]


@dataclass(frozen=True)
class EgoState:
    id: Hashable
    type: AgentType
    position: Vector2
    velocity: Vector2
    heading: float


@dataclass(frozen=True)
class ObjectState:
    id: Hashable
    type: AgentType
    position: Vector2
    velocity: Vector2
    heading: float


@dataclass(frozen=True)
class BEVFrame:
    frame_id: Hashable
    timestamp: float
    coordinate_frame: str
    ego: EgoState
    objects: Sequence[ObjectState]


@dataclass(frozen=True)
class _StateSample:
    type_index: int
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    heading: float


@dataclass(frozen=True)
class _BufferedFrame:
    frame_id: Hashable
    timestamp: float
    ego_id: Hashable
    ordered_ids: Tuple[Hashable, ...]
    states: Mapping[Hashable, _StateSample]


class HistoryBuffer:
    """Keep the latest fixed-rate BEV states and build deployment input tensors.

    Values are copied without an SE(2) transform. Consequently every buffered
    value must already use the same fixed Ego reference frame required by QCNet.
    """

    def __init__(
        self,
        max_history: int = 50,
        frequency_hz: float = 10.0,
        timestamp_tolerance_seconds: float = 0.03,
    ) -> None:
        if max_history != 50:
            raise ValueError('This checkpoint requires max_history=50.')
        if abs(float(frequency_hz) - 10.0) > 1e-6:
            raise ValueError('This checkpoint requires frequency_hz=10.0.')
        if timestamp_tolerance_seconds < 0:
            raise ValueError('timestamp_tolerance_seconds must be non-negative.')
        self.max_history = int(max_history)
        self.frequency_hz = float(frequency_hz)
        self.timestamp_tolerance_seconds = float(timestamp_tolerance_seconds)
        self._frames: Deque[_BufferedFrame] = deque(maxlen=self.max_history)
        self._ego_id: Optional[Hashable] = None

    @property
    def size(self) -> int:
        return len(self._frames)

    @property
    def ready(self) -> bool:
        return self.size == self.max_history

    def clear(self) -> None:
        self._frames.clear()
        self._ego_id = None

    @staticmethod
    def _type_index(value: AgentType) -> int:
        if isinstance(value, bool):
            raise ValueError('Agent type must be a name or integer index, not bool.')
        if isinstance(value, int):
            if 0 <= value < len(AGENT_TYPES):
                return value
            raise ValueError(f'Agent type index must be in [0, {len(AGENT_TYPES) - 1}].')
        if isinstance(value, str):
            normalized = value.strip().lower()
            try:
                return AGENT_TYPES.index(normalized)
            except ValueError as error:
                raise ValueError(
                    f'Unknown agent type {value!r}; expected one of {AGENT_TYPES}.') from error
        raise ValueError('Agent type must be a string name or integer index.')

    @staticmethod
    def _vector2(value: Vector2, field_name: str) -> Tuple[float, float]:
        try:
            tensor = torch.as_tensor(value, dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{field_name} must contain two numeric values.') from error
        if tuple(tensor.shape) != (2,):
            raise ValueError(f'{field_name} must have shape [2], got {tuple(tensor.shape)}.')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'{field_name} contains NaN or Inf.')
        return float(tensor[0]), float(tensor[1])

    @classmethod
    def _sample(cls, state: Union[EgoState, ObjectState], label: str) -> _StateSample:
        try:
            heading = float(state.heading)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{label}.heading must be numeric.') from error
        if not math.isfinite(heading):
            raise ValueError(f'{label}.heading contains NaN or Inf.')
        heading = math.atan2(math.sin(heading), math.cos(heading))
        return _StateSample(
            type_index=cls._type_index(state.type),
            position=cls._vector2(state.position, f'{label}.position'),
            velocity=cls._vector2(state.velocity, f'{label}.velocity'),
            heading=heading,
        )

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
            delta = timestamp - self._frames[-1].timestamp
            expected_delta = 1.0 / self.frequency_hz
            if delta <= 0:
                raise ValueError('BEVFrame timestamps must be strictly increasing.')
            if abs(delta - expected_delta) > self.timestamp_tolerance_seconds:
                raise ValueError(
                    f'Expected 10 Hz timestamps (delta {expected_delta:.3f}s), '
                    f'got {delta:.6f}s. Resample upstream before calling update().')

        ego_id = frame.ego.id
        if ego_id is None:
            raise ValueError('EgoState.id must not be None.')
        if self._ego_id is not None and ego_id != self._ego_id:
            raise ValueError('EgoState.id changed; call HistoryBuffer.clear() for a new stream.')

        states: Dict[Hashable, _StateSample] = {
            ego_id: self._sample(frame.ego, 'ego'),
        }
        ordered_ids: List[Hashable] = [ego_id]
        for index, obj in enumerate(frame.objects):
            if obj.id is None:
                raise ValueError(f'objects[{index}].id must not be None.')
            if obj.id in states:
                raise ValueError(f'Duplicate BEV object id: {obj.id!r}.')
            states[obj.id] = self._sample(obj, f'objects[{index}]')
            ordered_ids.append(obj.id)

        self._frames.append(_BufferedFrame(
            frame_id=frame.frame_id,
            timestamp=timestamp,
            ego_id=ego_id,
            ordered_ids=tuple(ordered_ids),
            states=states,
        ))
        self._ego_id = ego_id

    def to_qcnet_input(self, require_full_history: bool = True) -> Dict[str, object]:
        if not self._frames:
            raise RuntimeError('HistoryBuffer is empty.')
        if require_full_history and not self.ready:
            raise RuntimeError(
                f'HistoryBuffer needs {self.max_history} frames, currently has {self.size}.')

        frames = list(self._frames)
        latest = frames[-1]
        sample_counts = {
            agent_id: sum(agent_id in buffered_frame.states for buffered_frame in frames)
            for agent_id in latest.ordered_ids
        }
        if sample_counts[latest.ego_id] < 2:
            raise RuntimeError('The AV needs at least two valid samples before prediction.')
        # A brand-new track has no motion vector yet. Defer it until the second
        # valid sample instead of failing the complete real-time prediction.
        agent_ids = [
            agent_id for agent_id in latest.ordered_ids
            if sample_counts[agent_id] >= 2
        ]
        num_agents = len(agent_ids)
        history_offset = self.max_history - len(frames)
        position = torch.zeros(num_agents, self.max_history, 2, dtype=torch.float32)
        velocity = torch.zeros_like(position)
        heading = torch.zeros(num_agents, self.max_history, dtype=torch.float32)
        valid_mask = torch.zeros(num_agents, self.max_history, dtype=torch.bool)
        latest_types = [latest.states[agent_id].type_index for agent_id in agent_ids]

        for frame_index, buffered_frame in enumerate(frames, start=history_offset):
            for agent_index, agent_id in enumerate(agent_ids):
                sample = buffered_frame.states.get(agent_id)
                if sample is None:
                    continue
                if sample.type_index != latest_types[agent_index]:
                    raise ValueError(
                        f'Agent {agent_id!r} changed type inside the history window.')
                position[agent_index, frame_index] = torch.tensor(sample.position)
                velocity[agent_index, frame_index] = torch.tensor(sample.velocity)
                heading[agent_index, frame_index] = sample.heading
                valid_mask[agent_index, frame_index] = True

        return {
            'position': position,
            'velocity': velocity,
            'heading': heading,
            'valid_mask': valid_mask,
            'agent_type': torch.tensor(latest_types, dtype=torch.long),
            'av_index': agent_ids.index(latest.ego_id),
            'agent_ids': [str(agent_id) for agent_id in agent_ids],
            'coordinate_frame': 'ego',
            'frequency_hz': self.frequency_hz,
        }
