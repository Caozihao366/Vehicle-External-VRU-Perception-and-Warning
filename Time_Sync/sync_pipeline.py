"""
FR-01: 真正的多相机时间对齐

核心算法（与数据集无关，可复用到本地图/真机）：
  四路各自一条 (image, timestamp) 队列
  → 选基准时刻（默认 front 队列）
  → 在其余队列做最近邻匹配
  → 输出多帧组，并按毫秒阈值判定 available / degraded / unavailable

nuScenes 只负责「灌队列」，不再直接使用 sample 预绑定的多相机组。
"""

from __future__ import annotations

import json
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes

TZ_UTC8 = timezone(timedelta(hours=8))

DEFAULT_CAMERA_MAP = {
    "front": "CAM_FRONT",
    "left": "CAM_FRONT_LEFT",
    "right": "CAM_FRONT_RIGHT",
    "rear": "CAM_BACK",
}


# ======================================================================
# 通用数据结构 + 时间对齐算法（硬件无关）
# ======================================================================

@dataclass
class Frame:
    """一路相机上的一帧。真机/本地图只要填这三项即可。"""

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
            # 也要求每路相对基准的偏差不超过阈值
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
    """
    不依赖数据集的自检：人为制造时间偏移，验证最近邻是否找对。
    只打印到终端。
    """
    print("=" * 60)
    print("时间对齐算法自检（本地模拟队列，非 nuScenes）")
    base = 1_000_000_000_000
    streams = {
        "front": [
            Frame("front", "f0.jpg", base),
            Frame("front", "f1.jpg", base + 100_000),  # +100ms
        ],
        "left": [
            Frame("left", "l0.jpg", base + 8_000),  # +8ms → 应匹配 f0
            Frame("left", "l1.jpg", base + 108_000),
        ],
        "right": [
            Frame("right", "r0.jpg", base + 12_000),  # +12ms
            Frame("right", "r1.jpg", base + 112_000),
        ],
        "rear": [
            Frame("rear", "b0.jpg", base + 25_000),  # +25ms
            Frame("rear", "b1.jpg", base + 90_000),  # 更靠近 f1? 对 f0 仍应选 b0
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

    # 超阈值：rear 故意偏 80ms
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
# SyncPipeline：灌队列（nuScenes）+ 调用对齐器 + 组装附录 A JSON
# ======================================================================

class SyncPipeline:
    def __init__(
        self,
        dataroot: str = "./data",
        version: str = "v1.0-mini",
        output_json: str = "fr01_sync_output.json",
        sync_tolerance_ms: float = 35.0,
        camera_map: Optional[Dict[str, str]] = None,
        base_camera: str = "front",
        # keyframe≈0.5s一帧; interval=包内多帧; all≈相机全帧率
        ref_mode: str = "interval",
        ref_interval_ms: float = 100.0,
        verbose: bool = True,
    ) -> None:
        self.dataroot = dataroot
        self.version = version
        self.output_json = output_json
        self.sync_tolerance_ms = sync_tolerance_ms
        self.camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
        self.base_camera = base_camera
        self.ref_mode = ref_mode
        self.ref_interval_ms = ref_interval_ms
        self.max_pose_gap_s = 0.5
        self.verbose = verbose
        self.nusc: Optional[NuScenes] = None
        self.aligner = TemporalAligner(
            tolerance_ms=sync_tolerance_ms, base_camera=base_camera
        )

    def run(self, save: bool = True) -> List[dict]:
        self._ensure_nusc()
        assert self.nusc is not None

        if self.verbose:
            print("=" * 60)
            print("FR-01 真正时间对齐：独立队列 + 最近邻匹配")
            print(f"相机: {self.camera_map}")
            print(f"sync_tolerance_ms = {self.sync_tolerance_ms}")
            print(f"ref_mode = {self.ref_mode}", end="")
            if self.ref_mode == "interval":
                print(f"  ref_interval_ms = {self.ref_interval_ms}")
            else:
                print()
            print("=" * 60)

        outputs: List[dict] = []
        differ_from_bundle = 0
        compared_to_bundle = 0
        total_groups = 0
        qualities: List[str] = []
        spreads: List[float] = []
        ego_ok = 0

        for scene in self.nusc.scene:
            streams = self._build_camera_streams(scene)
            ref_times = self._select_ref_times(streams, scene)
            kin_ts, kin_speed, kin_yaw = self._scene_kinematics(scene)
            groups = self.aligner.align(streams, ref_times)

            keyframe_map = None
            if self.ref_mode == "keyframe":
                keyframe_map = {t: tok for t, tok in self._base_keyframe_refs(scene)}

            scene_out = []
            for group in groups:
                total_groups += 1
                if keyframe_map is not None:
                    sample_token = keyframe_map.get(group.ref_timestamp_us)
                    if sample_token:
                        compared_to_bundle += 1
                        if self._differs_from_sample_bundle(sample_token, group):
                            differ_from_bundle += 1
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
                n_key = len(self._base_keyframe_refs(scene))
                print(
                    f"{scene['name']}: queues={qlen}, "
                    f"keyframes={n_key}, aligned_groups={len(scene_out)}"
                )

        if save:
            out_path = Path(self.output_json)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)
            if self.verbose:
                self.print_report(
                    out_path, qualities, spreads, ego_ok,
                    differ_from_bundle, compared_to_bundle, total_groups,
                )
        elif self.verbose:
            self.print_report(
                Path(self.output_json), qualities, spreads, ego_ok,
                differ_from_bundle, compared_to_bundle, total_groups,
            )

        return outputs

    def print_report(
        self,
        out_path: Optional[Path],
        qualities: List[str],
        spreads: List[float],
        ego_ok: int = 0,
        differ_from_bundle: int = 0,
        compared_to_bundle: int = 0,
        total_groups: int = 0,
    ) -> None:
        status_count = {"available": 0, "degraded": 0, "unavailable": 0}
        for q in qualities:
            status_count[q] = status_count.get(q, 0) + 1

        n = len(qualities)
        spreads_arr = np.asarray(spreads, dtype=float) if spreads else np.array([0.0])
        avail_ratio = status_count.get("available", 0) / n if n else 0.0

        print("-" * 60)
        print("终端验收摘要（最近邻时间对齐）")
        print(f"  对齐算法       : 多路独立队列 + 基准时刻最近邻")
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
        if compared_to_bundle:
            print(f"  与 sample 预绑定不同: {differ_from_bundle}/{compared_to_bundle}")
        if out_path is not None:
            print(f"  JSON 路径      : {out_path.resolve()}")
        print("-" * 60)

    def _select_ref_times(
        self, streams: Dict[str, List[Frame]], scene: dict
    ) -> List[int]:
        """选择对齐基准时刻：决定每个 0.5s 窗口内取几组。"""
        if self.ref_mode == "keyframe":
            return [t for t, _ in self._base_keyframe_refs(scene)]

        base_q = streams[self.base_camera]
        if not base_q:
            return []
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

    # ------------------------------------------------------------------
    # nuScenes → 独立队列
    # ------------------------------------------------------------------

    def _ensure_nusc(self) -> None:
        if self.nusc is None:
            self.nusc = NuScenes(
                version=self.version, dataroot=self.dataroot, verbose=False
            )

    def _build_camera_streams(self, scene: dict) -> Dict[str, List[Frame]]:
        """每路相机沿 sample_data 链表收集全量帧（含非 keyframe），形成独立队列。"""
        assert self.nusc is not None
        sample0 = self.nusc.get("sample", scene["first_sample_token"])
        streams: Dict[str, List[Frame]] = {}

        for cam_key, channel in self.camera_map.items():
            sd_token = sample0["data"][channel]
            # 回到该相机链起点
            while True:
                sd = self.nusc.get("sample_data", sd_token)
                if sd["prev"] == "":
                    break
                sd_token = sd["prev"]

            frames: List[Frame] = []
            while sd_token != "":
                sd = self.nusc.get("sample_data", sd_token)
                frames.append(
                    Frame(
                        camera=cam_key,
                        path=sd["filename"],
                        timestamp_us=sd["timestamp"],
                        token=sd["token"],
                        meta={"is_key_frame": bool(sd["is_key_frame"])},
                    )
                )
                sd_token = sd["next"]

            frames.sort(key=lambda f: f.timestamp_us)
            streams[cam_key] = frames

        return streams

    def _base_keyframe_refs(self, scene: dict) -> List[Tuple[int, str]]:
        """返回 [(front_keyframe_timestamp_us, sample_token), ...]"""
        assert self.nusc is not None
        refs: List[Tuple[int, str]] = []
        sample_token = scene["first_sample_token"]
        channel = self.camera_map[self.base_camera]
        while sample_token != "":
            sample = self.nusc.get("sample", sample_token)
            sd = self.nusc.get("sample_data", sample["data"][channel])
            refs.append((sd["timestamp"], sample["token"]))
            sample_token = sample["next"]
        return refs

    def _differs_from_sample_bundle(self, sample_token: str, group: SyncGroup) -> bool:
        """最近邻选中的路径是否不同于 nuScenes sample 预绑定。"""
        assert self.nusc is not None
        sample = self.nusc.get("sample", sample_token)
        for cam_key, channel in self.camera_map.items():
            matched = group.frames.get(cam_key)
            if matched is None:
                return True
            bundled = self.nusc.get("sample_data", sample["data"][channel])
            if matched.path != bundled["filename"]:
                return True
        return False

    def _to_output_json(
        self,
        group: SyncGroup,
        kin_ts: np.ndarray,
        kin_speed: np.ndarray,
        kin_yaw: np.ndarray,
    ) -> dict:
        """附录 A 字段 + 各相机 frames(path/timestamp/delta_ms)。"""
        t_start = time.perf_counter()
        base_us = group.ref_timestamp_us
        ego = self._interpolate_ego(kin_ts, kin_speed, kin_yaw, base_us)

        quality = group.quality
        if ego is None:
            quality = "unavailable"
            ego = {"speed_kmh": None, "yaw_rate": None}

        frames_out: Dict[str, dict] = {}
        for logical_name, sensor_name in self.camera_map.items():
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

    def _scene_kinematics(
        self, scene: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._build_kinematics(self._collect_ego_poses(scene))

    def _collect_ego_poses(self, scene: dict) -> List[dict]:
        assert self.nusc is not None
        sample = self.nusc.get("sample", scene["first_sample_token"])
        sd_token = sample["data"]["LIDAR_TOP"]
        while True:
            sd = self.nusc.get("sample_data", sd_token)
            if sd["prev"] == "":
                break
            sd_token = sd["prev"]

        poses, seen = [], set()
        while sd_token != "":
            sd = self.nusc.get("sample_data", sd_token)
            ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
            if ep["token"] not in seen:
                seen.add(ep["token"])
                poses.append(ep)
            sd_token = sd["next"]
        poses.sort(key=lambda p: p["timestamp"])
        return poses

    def _build_kinematics(
        self, ego_poses: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(ego_poses) < 2:
            return np.array([]), np.array([]), np.array([])

        ts, speeds, yaw_rates = [], [], []
        for i in range(len(ego_poses) - 1):
            p0, p1 = ego_poses[i], ego_poses[i + 1]
            dt = (p1["timestamp"] - p0["timestamp"]) * 1e-6
            if dt <= 1e-6:
                continue
            dist = np.linalg.norm(
                np.array(p1["translation"]) - np.array(p0["translation"])
            )
            yaw0 = Quaternion(p0["rotation"]).yaw_pitch_roll[0]
            yaw1 = Quaternion(p1["rotation"]).yaw_pitch_roll[0]
            dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
            ts.append(p0["timestamp"])
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
        """ISO8601，含毫秒与 +08:00。"""
        dt = datetime.fromtimestamp(
            timestamp_us / 1e6, tz=timezone.utc
        ).astimezone(TZ_UTC8)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+08:00"


if __name__ == "__main__":
    demo_temporal_aligner(tolerance_ms=40.0)
    # keyframe≈0.5s一帧；interval=100ms → 每个标注间隔内约 5 组；all≈12Hz全量
    SyncPipeline(
        sync_tolerance_ms=50.0,
        ref_mode="interval",
        ref_interval_ms=5.0,
    ).run()
