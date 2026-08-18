"""FR-05 与上下游之间的稳定接口定义。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


BehaviorState = Literal[
    "normal",
    "running",
    "falling",
    "suspect_wrong_way",
    "wrong_way",
    "uncertain",
]
ModuleStatus = Literal["available", "degraded", "unavailable"]

REQUIRED_TRACK_FIELDS = (
    "track_id",
    "cls",
    "bbox",
    "bottom_center",
    "bbox_h",
    "speed_norm",
)


@dataclass(frozen=True)
class FR05Event:
    """单个异常行为事件，可直接序列化后提供给风险与日志模块。"""

    frame_id: int
    timestamp: float
    object_id: int | str
    class_id: int
    state: BehaviorState
    confidence: float
    source: str
    bbox_px: tuple[int, int, int, int]
    is_ghost: bool = False
    orient_face: str | None = None
    orient_confidence: float | None = None
    ego_lane_relation: str | None = None
    decision_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WrongWayDiagnostics:
    """外包逆行子模块的逐帧运行状态。"""

    orientation_enabled: bool
    lane_enabled: bool
    lane_ready: bool
    vehicle_count: int
    wrong_way_count: int
    suspect_count: int
    warnings: tuple[str, ...] = ()


@dataclass
class FR05Result:
    """FR-05 单帧输出；tracks 保持原对象引用以兼容现有主流程。"""

    frame_id: int
    timestamp: float
    tracks: list[dict[str, Any]]
    events: list[FR05Event] = field(default_factory=list)
    status: ModuleStatus = "available"
    latency_ms: float = 0.0
    capabilities: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    invalid_track_ids: list[int | str] = field(default_factory=list)

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "latency_ms": round(self.latency_ms, 3),
            "capabilities": dict(self.capabilities),
            "warnings": list(self.warnings),
            "invalid_track_ids": list(self.invalid_track_ids),
            "event_count": len(self.events),
        }


def validate_track(track: dict[str, Any]) -> list[str]:
    """返回轨迹接口错误列表；空列表表示满足 FR-05 最小输入契约。"""
    errors = [name for name in REQUIRED_TRACK_FIELDS if name not in track]
    bbox = track.get("bbox")
    if bbox is not None and (not hasattr(bbox, "__len__") or len(bbox) != 4):
        errors.append("bbox must contain four coordinates")
    bottom_center = track.get("bottom_center")
    if bottom_center is not None and (
        not hasattr(bottom_center, "__len__") or len(bottom_center) != 2
    ):
        errors.append("bottom_center must contain two coordinates")
    return errors
