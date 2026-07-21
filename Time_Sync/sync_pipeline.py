"""
FR-01: 多相机时间对齐

本文件只包含时间对齐与 JSON 输出，不读取任何数据集。
数据集读取器（例如 NuScenesDatasetSource）必须位于本文件之外，
并提供 iter_scenes() 返回统一的场景字典。
"""

from __future__ import annotations

import json
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pyquaternion import Quaternion

TZ_UTC8 = timezone(timedelta(hours=8))

# 逻辑名 → 交接 JSON 中的相机键（可改；默认沿用常见环视命名）
DEFAULT_OUTPUT_CAMERA_MAP = {
    "front": "CAM_FRONT",
    "left": "CAM_FRONT_LEFT",
    "right": "CAM_FRONT_RIGHT",
    "rear": "CAM_BACK",
}


# ======================================================================
# 时间对齐算法（硬件 / 数据集无关）
# ======================================================================

@dataclass
class Frame:
    """一路相机上的一帧。数据集读取函数只需提供 path / timestamp_us。"""

    camera: str
    path: str
    timestamp_us: int
    token: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class SyncGroup:
    """对齐后的多帧组（算法中间结果）。"""

    ref_timestamp_us: int
    frames: Dict[str, Optional[Frame]]
    delta_from_ref_ms: Dict[str, float]
    max_spread_ms: float
    quality: str  # available / degraded / unavailable


class TemporalAligner:
    """
    真正的时间对齐器：队列 + 基准时刻 + 最近邻。

    streams: { "front": [Frame,...], "left": [...], ... }  各队列须已按时间戳升序
    """

    def __init__(self, tolerance_ms: float = 35.0, base_camera: str = "front") -> None:
        self.tolerance_ms = tolerance_ms
        self.base_camera = base_camera

    @staticmethod
    def nearest(queue: Sequence[Frame], target_us: int) -> Tuple[Optional[Frame], float]:
        """在单路队列中找时间戳最接近 target 的帧，返回 (frame, |Δt| ms)。"""
        if not queue:
            return None, float("inf")

        ts = [f.timestamp_us for f in queue]
        i = bisect_left(ts, target_us)
        candidates: List[Frame] = []
        if i < len(queue):
            candidates.append(queue[i])
        if i > 0:
            candidates.append(queue[i - 1])

        best = min(candidates, key=lambda f: abs(f.timestamp_us - target_us))
        delta_ms = abs(best.timestamp_us - target_us) / 1000.0
        return best, delta_ms

    def align_one(
        self,
        streams: Dict[str, Sequence[Frame]],
        ref_timestamp_us: int,
    ) -> SyncGroup:
        """对单个基准时刻，在各队列中最近邻匹配。"""
        chosen: Dict[str, Optional[Frame]] = {}
        deltas: Dict[str, float] = {}

        for cam, queue in streams.items():
            frame, delta_ms = self.nearest(queue, ref_timestamp_us)
            chosen[cam] = frame
            deltas[cam] = delta_ms

        present = [f for f in chosen.values() if f is not None]
        if len(present) < len(streams) or any(f is None for f in chosen.values()):
            quality = "unavailable"
            max_spread = float("inf")
        else:
            us_list = [f.timestamp_us for f in present]
            max_spread = (max(us_list) - min(us_list)) / 1000.0
            max_delta = max(deltas.values()) if deltas else float("inf")
            if max_spread > self.tolerance_ms or max_delta > self.tolerance_ms:
                quality = "degraded"
            else:
                quality = "available"

        return SyncGroup(
            ref_timestamp_us=ref_timestamp_us,
            frames=chosen,
            delta_from_ref_ms={k: round(v, 3) for k, v in deltas.items()},
            max_spread_ms=max_spread if max_spread != float("inf") else max_spread,
            quality=quality,
        )

    def align(
        self,
        streams: Dict[str, Sequence[Frame]],
        ref_timestamps_us: Optional[Sequence[int]] = None,
    ) -> List[SyncGroup]:
        """
        批量对齐。
        ref_timestamps_us 为空时：用基准相机队列的每一帧时刻作为基准。
        """
        if self.base_camera not in streams:
            raise KeyError(f"缺少基准相机队列: {self.base_camera}")

        if ref_timestamps_us is None:
            ref_timestamps_us = [f.timestamp_us for f in streams[self.base_camera]]

        return [self.align_one(streams, t) for t in ref_timestamps_us]


def demo_temporal_aligner(tolerance_ms: float = 35.0) -> None:
    """不依赖数据集的自检：人造时间偏移，验证最近邻。"""
    print("=" * 60)
    print("时间对齐算法自检（模拟队列）")
    base = 1_000_000_000_000
    streams = {
        "front": [
            Frame("front", "f0.jpg", base),
            Frame("front", "f1.jpg", base + 100_000),
        ],
        "left": [
            Frame("left", "l0.jpg", base + 8_000),
            Frame("left", "l1.jpg", base + 108_000),
        ],
        "right": [
            Frame("right", "r0.jpg", base + 12_000),
            Frame("right", "r1.jpg", base + 112_000),
        ],
        "rear": [
            Frame("rear", "b0.jpg", base + 25_000),
            Frame("rear", "b1.jpg", base + 90_000),
        ],
    }
    aligner = TemporalAligner(tolerance_ms=tolerance_ms, base_camera="front")
    g0 = aligner.align_one(streams, base)
    ok = (
        g0.frames["front"].path == "f0.jpg"
        and g0.frames["left"].path == "l0.jpg"
        and g0.frames["right"].path == "r0.jpg"
        and g0.frames["rear"].path == "b0.jpg"
        and g0.quality == "available"
    )
    print(f"  基准 t0 匹配: {[g0.frames[c].path for c in ('front','left','right','rear')]}")
    print(f"  Δt(ms)      : {g0.delta_from_ref_ms}")
    print(f"  spread      : {g0.max_spread_ms:.3f} ms  quality={g0.quality}")
    print(f"  自检结果    : {'PASS' if ok else 'FAIL'}")

    streams_bad = {
        "front": [Frame("front", "f0.jpg", base)],
        "left": [Frame("left", "l0.jpg", base + 5_000)],
        "right": [Frame("right", "r0.jpg", base + 5_000)],
        "rear": [Frame("rear", "b0.jpg", base + 80_000)],
    }
    g1 = aligner.align_one(streams_bad, base)
    print(
        f"  超阈值用例  : spread={g1.max_spread_ms:.1f}ms → {g1.quality} "
        f"(期望 degraded)"
    )
    print("=" * 60)


# ======================================================================
# SyncPipeline：只依赖任意提供 iter_scenes() 的数据集读取器
# ======================================================================

class SyncPipeline:
    def __init__(
        self,
        source: Any,
        output_json: str = "fr01_sync_output.json",
        sync_tolerance_ms: float = 35.0,
        output_camera_map: Optional[Dict[str, str]] = None,
        base_camera: str = "front",
        # keyframe=meta.is_key_frame; interval=包内抽样; all=基准相机全帧
        ref_mode: str = "interval",
        ref_interval_ms: float = 100.0,
        require_ego: bool = False,
        verbose: bool = True,
    ) -> None:
        self.source = source
        self.output_json = output_json
        self.sync_tolerance_ms = sync_tolerance_ms
        self.output_camera_map = dict(
            output_camera_map or DEFAULT_OUTPUT_CAMERA_MAP
        )
        self.base_camera = base_camera
        self.ref_mode = ref_mode
        self.ref_interval_ms = ref_interval_ms
        self.require_ego = require_ego
        self.max_pose_gap_s = 0.5
        self.verbose = verbose
        self.aligner = TemporalAligner(
            tolerance_ms=sync_tolerance_ms, base_camera=base_camera
        )

    def run(self, save: bool = True) -> List[dict]:
        if self.verbose:
            print("=" * 60)
            print("FR-01 时间对齐：数据集读取器 → 最近邻匹配")
            print(f"source = {type(self.source).__name__}")
            print(f"output_cameras = {self.output_camera_map}")
            print(f"sync_tolerance_ms = {self.sync_tolerance_ms}")
            print(f"ref_mode = {self.ref_mode}", end="")
            if self.ref_mode == "interval":
                print(f"  ref_interval_ms = {self.ref_interval_ms}")
            else:
                print()
            print("=" * 60)

        outputs: List[dict] = []
        total_groups = 0
        qualities: List[str] = []
        spreads: List[float] = []
        ego_ok = 0

        for scene in self.source.iter_scenes():
            streams = self._to_frame_streams(scene["streams"])
            ref_times = self._select_ref_times(streams)
            kin_ts, kin_speed, kin_yaw = self._build_kinematics(
                scene.get("ego_poses") or []
            )
            groups = self.aligner.align(streams, ref_times)

            scene_out = []
            for group in groups:
                total_groups += 1
                item = self._to_output_json(group, kin_ts, kin_speed, kin_yaw)
                scene_out.append(item)
                qualities.append(item["sensor_availability"])
                if group.max_spread_ms != float("inf"):
                    spreads.append(group.max_spread_ms)
                if item["ego_status"]["speed_kmh"] is not None:
                    ego_ok += 1

            outputs.extend(scene_out)
            if self.verbose:
                qlen = {k: len(v) for k, v in streams.items()}
                print(
                    f"{scene.get('name', 'unnamed')}: "
                    f"queues={qlen}, aligned_groups={len(scene_out)}"
                )

        if save:
            out_path = Path(self.output_json)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)
            if self.verbose:
                self.print_report(out_path, qualities, spreads, ego_ok, total_groups)
        elif self.verbose:
            self.print_report(
                Path(self.output_json), qualities, spreads, ego_ok, total_groups
            )

        return outputs

    def print_report(
        self,
        out_path: Optional[Path],
        qualities: List[str],
        spreads: List[float],
        ego_ok: int = 0,
        total_groups: int = 0,
    ) -> None:
        status_count = {"available": 0, "degraded": 0, "unavailable": 0}
        for q in qualities:
            status_count[q] = status_count.get(q, 0) + 1

        n = len(qualities) or total_groups
        spreads_arr = np.asarray(spreads, dtype=float) if spreads else np.array([0.0])
        avail_ratio = status_count.get("available", 0) / n if n else 0.0

        print("-" * 60)
        print("终端验收摘要（最近邻时间对齐）")
        print(f"  对齐算法       : 多路独立队列 + 基准时刻最近邻")
        print(f"  输入接口       : {type(self.source).__name__}")
        print(f"  抽样模式       : {self.ref_mode}", end="")
        if self.ref_mode == "interval":
            print(f" @ {self.ref_interval_ms} ms")
        else:
            print()
        print(f"  容忍度门限     : {self.sync_tolerance_ms} ms")
        print(f"  总帧组         : {n}")
        print(
            f"  availability   : available={status_count.get('available', 0)}, "
            f"degraded={status_count.get('degraded', 0)}, "
            f"unavailable={status_count.get('unavailable', 0)}"
        )
        print(f"  available 比例 : {avail_ratio * 100:.1f}%")
        print(f"  ego 插值成功   : {ego_ok}/{n}")
        print(
            f"  四路跨度 ms    : P50={float(np.percentile(spreads_arr, 50)):.3f}, "
            f"P99={float(np.percentile(spreads_arr, 99)):.3f}, "
            f"Max={float(spreads_arr.max()):.3f}"
        )
        if out_path is not None:
            print(f"  JSON 路径      : {out_path.resolve()}")
        print("-" * 60)

    def _select_ref_times(self, streams: Dict[str, List[Frame]]) -> List[int]:
        base_q = streams.get(self.base_camera, [])
        if not base_q:
            return []

        if self.ref_mode == "keyframe":
            keyed = [
                f.timestamp_us
                for f in base_q
                if f.meta.get("is_key_frame")
            ]
            return keyed if keyed else [f.timestamp_us for f in base_q]

        if self.ref_mode == "all":
            return [f.timestamp_us for f in base_q]

        if self.ref_mode != "interval":
            raise ValueError(
                f"未知 ref_mode={self.ref_mode!r}，可选 keyframe|interval|all"
            )

        refs: List[int] = []
        next_t = base_q[0].timestamp_us
        step = int(self.ref_interval_ms * 1000)
        for fr in base_q:
            if fr.timestamp_us >= next_t:
                refs.append(fr.timestamp_us)
                next_t = fr.timestamp_us + step
        return refs

    @staticmethod
    def _to_frame_streams(
        raw_streams: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> Dict[str, List[Frame]]:
        """把读取器输出的标准字典转换为对齐算法所需的 Frame 队列。"""
        streams: Dict[str, List[Frame]] = {}
        for camera, entries in raw_streams.items():
            frames: List[Frame] = []
            for index, entry in enumerate(entries):
                if "path" not in entry or "timestamp_us" not in entry:
                    raise ValueError(
                        f"camera={camera!r} index={index}: "
                        "每帧必须提供 path 与 timestamp_us"
                    )
                frames.append(
                    Frame(
                        camera=camera,
                        path=str(entry["path"]),
                        timestamp_us=int(entry["timestamp_us"]),
                        token=str(entry.get("token", "")),
                        meta=dict(entry.get("meta") or {}),
                    )
                )
            streams[camera] = sorted(frames, key=lambda frame: frame.timestamp_us)
        return streams

    def _to_output_json(
        self,
        group: SyncGroup,
        kin_ts: np.ndarray,
        kin_speed: np.ndarray,
        kin_yaw: np.ndarray,
    ) -> dict:
        t_start = time.perf_counter()
        base_us = group.ref_timestamp_us
        has_ego_track = len(kin_ts) > 0
        ego = self._interpolate_ego(kin_ts, kin_speed, kin_yaw, base_us)

        quality = group.quality
        if ego is None:
            if self.require_ego or has_ego_track:
                # 有轨迹却插值失败，或调用方强制要求 ego → unavailable
                quality = "unavailable"
            ego = {"speed_kmh": None, "yaw_rate": None}

        frames_out: Dict[str, dict] = {}
        for logical_name, sensor_name in self.output_camera_map.items():
            fr = group.frames.get(logical_name)
            if fr is None:
                continue
            frames_out[sensor_name] = {
                "path": fr.path,
                "timestamp": self._us_to_iso8601(fr.timestamp_us),
                "delta_ms": round(
                    float(group.delta_from_ref_ms.get(logical_name, 0.0)), 1
                ),
            }

        process_ms = (time.perf_counter() - t_start) * 1000.0
        spread = (
            group.max_spread_ms if group.max_spread_ms != float("inf") else 0.0
        )
        total_delay_ms = float(max(spread, process_ms))

        return {
            "timestamp": self._us_to_iso8601(base_us),
            "ego_status": {
                "speed_kmh": None
                if ego["speed_kmh"] is None
                else round(float(ego["speed_kmh"]), 3),
                "yaw_rate": None
                if ego["yaw_rate"] is None
                else round(float(ego["yaw_rate"]), 6),
            },
            "frames": frames_out,
            "sensor_availability": quality,
            "vru_objects": [],
            "performance": {
                "total_delay_ms": round(total_delay_ms, 3),
            },
        }

    def _build_kinematics(
        self, ego_poses: Sequence[Mapping[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(ego_poses) < 2:
            return np.array([]), np.array([]), np.array([])

        ts, speeds, yaw_rates = [], [], []
        for i in range(len(ego_poses) - 1):
            p0, p1 = ego_poses[i], ego_poses[i + 1]
            dt = (int(p1["timestamp_us"]) - int(p0["timestamp_us"])) * 1e-6
            if dt <= 1e-6:
                continue
            dist = np.linalg.norm(
                np.asarray(p1["translation"], dtype=float)
                - np.asarray(p0["translation"], dtype=float)
            )
            yaw0 = Quaternion(list(p0["rotation"])).yaw_pitch_roll[0]
            yaw1 = Quaternion(list(p1["rotation"])).yaw_pitch_roll[0]
            dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
            ts.append(int(p0["timestamp_us"]))
            speeds.append(dist / dt * 3.6)
            yaw_rates.append(dyaw / dt)
        return (
            np.asarray(ts, dtype=np.int64),
            np.asarray(speeds),
            np.asarray(yaw_rates),
        )

    def _interpolate_ego(
        self,
        ts_us: np.ndarray,
        speeds: np.ndarray,
        yaw_rates: np.ndarray,
        target_us: int,
    ) -> Optional[Dict[str, float]]:
        if len(ts_us) == 0:
            return None
        gap = int(self.max_pose_gap_s * 1e6)
        if target_us < ts_us[0]:
            return (
                {"speed_kmh": float(speeds[0]), "yaw_rate": float(yaw_rates[0])}
                if ts_us[0] - target_us <= gap
                else None
            )
        if target_us > ts_us[-1]:
            return (
                {"speed_kmh": float(speeds[-1]), "yaw_rate": float(yaw_rates[-1])}
                if target_us - ts_us[-1] <= gap
                else None
            )
        idx = int(np.searchsorted(ts_us, target_us, side="right") - 1)
        idx = max(0, min(idx, len(ts_us) - 1))
        if idx >= len(ts_us) - 1:
            return {"speed_kmh": float(speeds[idx]), "yaw_rate": float(yaw_rates[idx])}
        t0, t1 = int(ts_us[idx]), int(ts_us[idx + 1])
        if (t1 - t0) * 1e-6 > self.max_pose_gap_s:
            return None
        a = 0.0 if t1 == t0 else float(np.clip((target_us - t0) / (t1 - t0), 0, 1))
        return {
            "speed_kmh": float(speeds[idx] * (1 - a) + speeds[idx + 1] * a),
            "yaw_rate": float(yaw_rates[idx] * (1 - a) + yaw_rates[idx + 1] * a),
        }

    @staticmethod
    def _us_to_iso8601(timestamp_us: int) -> str:
        dt = datetime.fromtimestamp(
            timestamp_us / 1e6, tz=timezone.utc
        ).astimezone(TZ_UTC8)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+08:00"

