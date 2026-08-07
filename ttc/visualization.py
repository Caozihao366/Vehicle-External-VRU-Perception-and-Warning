"""Optional camera/BEV visualization for offline TTC video and demonstrations.

The module is not imported by the TTC package root. Calculation-only users do
not load OpenCV. Inputs are generic TTC trajectories/results, never dataset
records, ground-truth trajectories, tokens, or annotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .models import (
    EgoTrajectoryPoint,
    MultimodalTTCResult,
    PredictionMode,
    VRUTrajectoryPoint,
)

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


RISK_BGR = {
    "safe": (40, 190, 40),
    "notice": (0, 220, 255),
    "warning": (0, 140, 255),
    "danger": (0, 0, 255),
}
RISK_RANK = {"safe": 0, "notice": 1, "warning": 2, "danger": 3}


@dataclass(frozen=True)
class VisualizationTarget:
    """Display input for one evaluated VRU.

    image_point is an optional camera pixel supplied by perception/BEV matching.
    This module intentionally performs no calibration or camera projection.
    """

    track_id: int | str
    category: str
    modes: tuple[PredictionMode, ...]
    result: MultimodalTTCResult
    image_point: tuple[int, int] | None = None


@dataclass(frozen=True)
class VisualizationConfig:
    """Display-only values; none changes the TTC calculation."""

    x_range_m: tuple[float, float] = (-5.0, 35.0)
    y_range_m: tuple[float, float] = (-15.0, 15.0)
    bev_width_px: int = 1000
    panel_height_px: int = 900
    camera_width_px: int = 1200
    grid_step_m: float = 5.0
    ego_length_m: float = 4.8
    ego_width_m: float = 2.0
    safety_margin_m: float = 0.4
    max_table_rows: int = 10
    show_all_modes: bool = True
    show_uncertainty: bool = True
    title: str = "VRU TTC"

    def __post_init__(self) -> None:
        if self.x_range_m[1] <= self.x_range_m[0]:
            raise ValueError("x_range_m must be increasing")
        if self.y_range_m[1] <= self.y_range_m[0]:
            raise ValueError("y_range_m must be increasing")
        if min(self.bev_width_px, self.panel_height_px, self.camera_width_px) <= 0:
            raise ValueError("pixel dimensions must be positive")
        if self.grid_step_m <= 0:
            raise ValueError("grid_step_m must be positive")
        if min(self.ego_length_m, self.ego_width_m) <= 0:
            raise ValueError("ego dimensions must be positive")
        if self.safety_margin_m < 0 or self.max_table_rows < 1:
            raise ValueError("margin and table row count are invalid")


def _require_backend() -> None:
    if cv2 is None or np is None:
        raise ImportError(
            "Visualization requires opencv-python and numpy; "
            "TTC calculation itself does not require this optional module"
        )


def rank_targets(
    targets: Sequence[VisualizationTarget],
) -> tuple[VisualizationTarget, ...]:
    """Sort by risk, earlier TTC, smaller clearance, and stable track id."""

    def key(target: VisualizationTarget) -> tuple[float, float, float, str]:
        result = target.result
        ttc = math.inf if result.earliest_ttc_ms is None else result.earliest_ttc_ms
        return (
            -RISK_RANK.get(result.risk_level, -1),
            ttc,
            result.min_clearance_m,
            str(target.track_id),
        )

    return tuple(sorted(targets, key=key))


def _pixel(x_m: float, y_m: float, cfg: VisualizationConfig) -> tuple[int, int]:
    x0, x1 = cfg.x_range_m
    y0, y1 = cfg.y_range_m
    u = int(round((y1 - y_m) / (y1 - y0) * (cfg.bev_width_px - 1)))
    v = int(round((x1 - x_m) / (x1 - x0) * (cfg.panel_height_px - 1)))
    return u, v


def _polyline(
    image: Any,
    points: Sequence[EgoTrajectoryPoint] | Sequence[VRUTrajectoryPoint],
    color: tuple[int, int, int],
    thickness: int,
    cfg: VisualizationConfig,
) -> None:
    pixels = [_pixel(point.x, point.y, cfg) for point in points]
    if len(pixels) >= 2:
        cv2.polylines(
            image, [np.asarray(pixels, dtype=np.int32)], False,
            color, thickness, cv2.LINE_AA,
        )
    elif pixels:
        cv2.circle(image, pixels[0], max(2, thickness), color, -1, cv2.LINE_AA)


def _grid(bev: Any, cfg: VisualizationConfig) -> None:
    x0, x1 = cfg.x_range_m
    y0, y1 = cfg.y_range_m
    x = math.ceil(x0 / cfg.grid_step_m) * cfg.grid_step_m
    while x <= x1 + 1e-9:
        cv2.line(bev, _pixel(x, y0, cfg), _pixel(x, y1, cfg), (55, 55, 55), 1)
        x += cfg.grid_step_m
    y = math.ceil(y0 / cfg.grid_step_m) * cfg.grid_step_m
    while y <= y1 + 1e-9:
        cv2.line(bev, _pixel(x0, y, cfg), _pixel(x1, y, cfg), (55, 55, 55), 1)
        y += cfg.grid_step_m


def _ego_envelope(
    bev: Any,
    trajectory: Sequence[EgoTrajectoryPoint],
    cfg: VisualizationConfig,
) -> None:
    length = cfg.ego_length_m + 2.0 * cfg.safety_margin_m
    width = cfg.ego_width_m + 2.0 * cfg.safety_margin_m
    stride = max(1, len(trajectory) // 15)
    sampled = list(trajectory[::stride])
    if trajectory and sampled[-1] is not trajectory[-1]:
        sampled.append(trajectory[-1])
    local = (
        (length / 2, width / 2), (length / 2, -width / 2),
        (-length / 2, -width / 2), (-length / 2, width / 2),
    )
    for point in sampled:
        c, s = math.cos(point.yaw), math.sin(point.yaw)
        polygon = [
            _pixel(
                point.x + c * lx - s * ly,
                point.y + s * lx + c * ly,
                cfg,
            )
            for lx, ly in local
        ]
        cv2.polylines(
            bev, [np.asarray(polygon, dtype=np.int32)], True,
            (150, 85, 25), 1, cv2.LINE_AA,
        )
    _polyline(bev, trajectory, (255, 155, 45), 5, cfg)


def _selected_mode(target: VisualizationTarget) -> PredictionMode | None:
    if not target.modes:
        return None
    preferred = target.result.critical_mode_id or target.result.most_likely_mode_id
    return next(
        (mode for mode in target.modes if mode.mode_id == preferred),
        max(target.modes, key=lambda mode: mode.probability),
    )


def _uncertainty(
    bev: Any,
    point: VRUTrajectoryPoint,
    color: tuple[int, int, int],
    cfg: VisualizationConfig,
) -> None:
    if point.sigma_x_m <= 0 and point.sigma_y_m <= 0:
        return
    x_scale = (cfg.panel_height_px - 1) / (cfg.x_range_m[1] - cfg.x_range_m[0])
    y_scale = (cfg.bev_width_px - 1) / (cfg.y_range_m[1] - cfg.y_range_m[0])
    axes = (
        max(1, round(point.sigma_y_m * y_scale)),
        max(1, round(point.sigma_x_m * x_scale)),
    )
    cv2.ellipse(bev, _pixel(point.x, point.y, cfg), axes, 0, 0, 360, color, 1)


def _camera_panel(frame: Any, cfg: VisualizationConfig) -> tuple[Any, float, int, int]:
    if frame is None:
        blank = np.full(
            (cfg.panel_height_px, cfg.camera_width_px, 3), 22, dtype=np.uint8
        )
        return blank, 1.0, 0, 0
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError("camera_frame must be a BGR image array or None")
    scale = min(
        cfg.camera_width_px / frame.shape[1],
        cfg.panel_height_px / frame.shape[0],
    )
    size = (
        max(1, round(frame.shape[1] * scale)),
        max(1, round(frame.shape[0] * scale)),
    )
    resized = cv2.resize(
        frame, size,
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    panel = np.full(
        (cfg.panel_height_px, cfg.camera_width_px, 3), 22, dtype=np.uint8
    )
    offset_x = (cfg.camera_width_px - resized.shape[1]) // 2
    offset_y = (cfg.panel_height_px - resized.shape[0]) // 2
    panel[
        offset_y : offset_y + resized.shape[0],
        offset_x : offset_x + resized.shape[1],
    ] = resized
    return panel, scale, offset_x, offset_y


def render_ttc_frame(
    camera_frame: Any,
    ego_trajectory: Sequence[EgoTrajectoryPoint],
    targets: Sequence[VisualizationTarget],
    *,
    config: VisualizationConfig | None = None,
    frame_index: int | None = None,
    timestamp_ms: int | None = None,
) -> Any:
    """Render one camera plus metric-BEV frame without changing TTC results."""

    _require_backend()
    cfg = config or VisualizationConfig()
    if not ego_trajectory:
        raise ValueError("ego_trajectory must not be empty")
    ranked = rank_targets(targets)
    camera, camera_scale, offset_x, offset_y = _camera_panel(camera_frame, cfg)
    bev = np.full(
        (cfg.panel_height_px, cfg.bev_width_px, 3), 28, dtype=np.uint8
    )
    _grid(bev, cfg)
    _ego_envelope(bev, ego_trajectory, cfg)

    for rank, target in enumerate(ranked, 1):
        result = target.result
        color = RISK_BGR.get(result.risk_level, (180, 180, 180))
        selected = _selected_mode(target)
        if cfg.show_all_modes:
            for mode in target.modes:
                if mode is not selected:
                    _polyline(bev, mode.points, color, 1, cfg)
        if selected is not None:
            _polyline(bev, selected.points, color, 5 if rank == 1 else 3, cfg)
            if selected.points:
                start = _pixel(selected.points[0].x, selected.points[0].y, cfg)
                cv2.circle(bev, start, 11 if rank == 1 else 7, color, -1)
                if rank == 1:
                    cv2.circle(bev, start, 17, (255, 255, 255), 3)
                cv2.putText(
                    bev, f"#{rank}", (start[0] + 10, start[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2,
                )
                if cfg.show_uncertainty:
                    uncertainty_stride = max(1, len(selected.points) // 5)
                    uncertainty_points = list(
                        selected.points[::uncertainty_stride]
                    )
                    if uncertainty_points[-1] is not selected.points[-1]:
                        uncertainty_points.append(selected.points[-1])
                    for point in uncertainty_points:
                        _uncertainty(bev, point, color, cfg)

        collision = next(
            (
                mode for mode in result.mode_results
                if mode.mode_id == result.critical_mode_id
            ),
            None,
        )
        if (
            collision is not None
            and collision.collision_x_m is not None
            and collision.collision_y_m is not None
        ):
            cv2.drawMarker(
                bev,
                _pixel(collision.collision_x_m, collision.collision_y_m, cfg),
                color, cv2.MARKER_TILTED_CROSS, 24, 3,
            )

        if target.image_point is not None and camera_frame is not None:
            u = round(target.image_point[0] * camera_scale) + offset_x
            v = round(target.image_point[1] * camera_scale) + offset_y
            if 0 <= u < camera.shape[1] and 0 <= v < camera.shape[0]:
                cv2.circle(camera, (u, v), 18 if rank == 1 else 11, color, 5 if rank == 1 else 2)
                cv2.putText(
                    camera, f"#{rank} ID:{target.track_id}",
                    (u + 12, max(24, v - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2,
                )

    suffix = []
    if frame_index is not None:
        suffix.append(f"FRAME {frame_index}")
    if timestamp_ms is not None:
        suffix.append(f"T={timestamp_ms} ms")
    heading = cfg.title + ((" | " + " | ".join(suffix)) if suffix else "")
    cv2.putText(camera, heading, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2)
    cv2.putText(
        camera, f"Evaluated VRUs: {len(ranked)} | white ring = highest risk",
        (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2,
    )

    source = ranked[0].result.ego_trajectory_source if ranked else "planner/ctrv"
    cv2.putText(
        bev, f"TTC BEV | ego source: {source}", (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2,
    )
    cv2.putText(
        bev, "Rank  ID       Type       Risk     TTC(ms)  Clear(m)  Prob",
        (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (225, 225, 225), 1,
    )
    for row, target in enumerate(ranked[: cfg.max_table_rows], 1):
        result = target.result
        ttc = "--" if result.earliest_ttc_ms is None else str(result.earliest_ttc_ms)
        line = (
            f"#{row:<3} {str(target.track_id)[:8]:<8} "
            f"{target.category[:10]:<10} {result.risk_level:<8} "
            f"{ttc:<8} {result.min_clearance_m:7.2f} "
            f"{result.collision_probability:4.2f}"
        )
        cv2.putText(
            bev, line, (18, 66 + row * 27), cv2.FONT_HERSHEY_SIMPLEX,
            0.46, RISK_BGR.get(result.risk_level, (180, 180, 180)), 1,
        )
    if len(ranked) > cfg.max_table_rows:
        cv2.putText(
            bev, f"+ {len(ranked) - cfg.max_table_rows} lower-risk targets drawn",
            (18, 94 + cfg.max_table_rows * 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1,
        )
    cv2.putText(
        bev, "Blue: planner/CTRV ego envelope | colored: model trajectories",
        (18, cfg.panel_height_px - 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1,
    )
    cv2.putText(
        bev, "X: predicted collision | ellipse: model uncertainty | no ground truth",
        (18, cfg.panel_height_px - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1,
    )
    return np.hstack((camera, bev))


class TTCVideoWriter:
    """Lazy MP4 writer for rendered frames."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        fps: float,
        codec: str = "mp4v",
    ) -> None:
        _require_backend()
        if fps <= 0:
            raise ValueError("fps must be positive")
        if len(codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        self.output_path = Path(output_path)
        self.fps = float(fps)
        self.codec = codec
        self._writer = None
        self._frame_size: tuple[int, int] | None = None

    def write(self, frame: Any) -> None:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a BGR image array")
        frame_size = (frame.shape[1], frame.shape[0])
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*self.codec),
                self.fps,
                frame_size,
            )
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise OSError(f"cannot create video: {self.output_path}")
            self._frame_size = frame_size
        elif frame_size != self._frame_size:
            raise ValueError(
                f"frame size changed from {self._frame_size} to {frame_size}"
            )
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "TTCVideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
