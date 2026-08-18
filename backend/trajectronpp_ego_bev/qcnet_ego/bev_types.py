"""Public Python object types produced by the real-time BEV module."""

from dataclasses import dataclass
from typing import List, Union


@dataclass
class EgoState:
    """Current ego-vehicle state."""

    id: str
    position: List[float]
    velocity: List[float]
    heading: float


@dataclass
class ObjectState:
    """Current state of one tracked object."""

    id: Union[str, int]
    type: str
    position: List[float]
    velocity: List[float]
    heading: float


@dataclass
class BEVFrame:
    """One 10 Hz frame from the BEV perception/tracking module."""

    frame_id: int
    timestamp: float
    coordinate_frame: str
    ego: EgoState
    objects: List[ObjectState]
