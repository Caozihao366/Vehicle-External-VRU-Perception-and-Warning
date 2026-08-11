"""Risk classification and per-track alarm hysteresis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _RiskMemory:
    level: str
    lower_count: int = 0
    danger_hold_remaining: int = 0


class RiskStateMachine:
    """Per-track hysteresis: immediate upgrades and confirmed downgrades."""

    LEVELS = ("safe", "notice", "warning", "danger")

    def __init__(self, downgrade_confirm_frames: int = 3, danger_hold_frames: int = 5):
        if downgrade_confirm_frames < 1 or danger_hold_frames < 0:
            raise ValueError("risk state-machine frame counts are invalid")
        self.downgrade_confirm_frames = downgrade_confirm_frames
        self.danger_hold_frames = danger_hold_frames
        self._tracks: dict[int | str, _RiskMemory] = {}

    def update(self, track_id: int | str, raw_level: str) -> str:
        if raw_level not in self.LEVELS:
            raise ValueError(f"unknown risk level: {raw_level}")
        memory = self._tracks.get(track_id)
        if memory is None:
            memory = _RiskMemory(
                level=raw_level,
                danger_hold_remaining=(
                    self.danger_hold_frames if raw_level == "danger" else 0
                ),
            )
            self._tracks[track_id] = memory
            return memory.level

        raw_rank = self.LEVELS.index(raw_level)
        current_rank = self.LEVELS.index(memory.level)
        if raw_rank > current_rank:
            memory.level = raw_level
            memory.lower_count = 0
            memory.danger_hold_remaining = (
                self.danger_hold_frames if raw_level == "danger" else 0
            )
        elif raw_rank == current_rank:
            memory.lower_count = 0
            if memory.level == "danger":
                memory.danger_hold_remaining = self.danger_hold_frames
        elif memory.level == "danger" and memory.danger_hold_remaining > 0:
            memory.danger_hold_remaining -= 1
            memory.lower_count = 0
        else:
            memory.lower_count += 1
            if memory.lower_count >= self.downgrade_confirm_frames:
                memory.level = raw_level
                memory.lower_count = 0
                memory.danger_hold_remaining = 0
        return memory.level

    def clear(self, track_id: int | str) -> None:
        self._tracks.pop(track_id, None)

    def clear_all(self) -> None:
        self._tracks.clear()


def classify_risk(
    earliest_ttc_s: float | None,
    collision_probability: float,
    *,
    behavior_state: str = "normal",
    danger_ttc_s: float = 1.2,
    warning_ttc_s: float = 2.5,
) -> tuple[str, str]:
    """Return the document-defined risk enum and its primary reason."""
    allowed_behaviors = {"normal", "running", "falling", "wrong_way", "uncertain"}
    if behavior_state not in allowed_behaviors:
        raise ValueError(
            f"behavior_state must be one of {', '.join(sorted(allowed_behaviors))}"
        )
    if danger_ttc_s <= 0 or warning_ttc_s <= danger_ttc_s:
        raise ValueError("risk TTC thresholds are invalid")

    if earliest_ttc_s is None:
        if behavior_state == "uncertain":
            return "notice", "uncertain_intent"
        return "safe", "no_collision_within_horizon"
    if collision_probability < 0.05:
        return "notice", "low_probability_collision_mode"
    if (
        behavior_state == "falling"
        and earliest_ttc_s <= warning_ttc_s
        and collision_probability >= 0.10
    ):
        return "danger", "falling_behavior_with_near_path_conflict"

    # Dangerous behavior advances warning time without modifying physical TTC.
    behavior_factor = {
        "normal": 1.0,
        "running": 1.25,
        "falling": 1.0,
        "wrong_way": 1.15,
        "uncertain": 1.10,
    }[behavior_state]
    if (
        earliest_ttc_s <= danger_ttc_s * behavior_factor
        and collision_probability >= 0.20
    ):
        return "danger", f"{behavior_state}_short_ttc"
    if (
        earliest_ttc_s <= warning_ttc_s * behavior_factor
        and collision_probability >= 0.10
    ):
        return "warning", f"{behavior_state}_trajectory_conflict"
    return "notice", "predicted_path_conflict"

