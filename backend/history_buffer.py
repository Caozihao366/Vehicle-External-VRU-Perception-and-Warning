"""
QCNet 历史缓冲区
================
将逐帧 BEVFrame 累积为 QCNet 所需的 [N, 50, 2] 历史张量。
处理目标出生/消失 (valid_mask) 和自车 agent 插入。
"""
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from bev_types import BEVFrame
from system_config import AGENT_TYPE_MAP, QCNET_HISTORY_STEPS


class HistoryBuffer:
    """
    累积 BEVFrame 历史，构建 QCNet 输入。

    用法:
        buf = HistoryBuffer()
        buf.append(bev_frame)
        if buf.ready():
            qcnet_input = buf.build_qcnet_input()
    """

    def __init__(self, history_steps: int = QCNET_HISTORY_STEPS):
        self.history_steps = history_steps
        self._frames: deque = deque(maxlen=history_steps)
        # 每个 track_id 的 agent_type
        self._agent_types: Dict = {}

    def append(self, frame: BEVFrame):
        """追加一帧 BEVFrame。"""
        self._frames.append(frame)
        for obj in frame.objects:
            if obj.id not in self._agent_types:
                self._agent_types[obj.id] = self._type_to_int(obj.type)

    def ready(self, min_steps: int = None) -> bool:
        """历史帧数是否足够。"""
        min_steps = min_steps or self.history_steps
        return len(self._frames) >= min_steps

    def __len__(self):
        return len(self._frames)

    @staticmethod
    def _type_to_int(type_str: str) -> int:
        """bev_types type 字符串 → QCNet agent_type 索引。"""
        mapping = {
            "vehicle": 0,
            "pedestrian": 1,
            "motorcyclist": 2,
            "cyclist": 3,
            "bus": 4,
        }
        return mapping.get(type_str, 9)  # 9=unknown

    def build_qcnet_input(
        self,
        *,
        ego_speed_mps: float = 0.0,
        ego_yaw_rate_rps: float = 0.0,
        output_frequency_hz: float = 10.0,
    ) -> Optional[dict]:
        """构建最新时刻固定 ego 坐标系下的 QCNet 历史输入。"""
        if len(self._frames) < 2 or output_frequency_hz <= 0:
            return None

        latest = self._frames[-1]
        latest_t = float(latest.timestamp)
        all_ids = ["ego"]
        latest_type = {}
        for obj in latest.objects:
            sid = str(obj.id)
            if sid not in all_ids:
                all_ids.append(sid)
                latest_type[sid] = obj.type

        n = len(all_ids)
        h = self.history_steps
        if n < 2:
            return None

        grid = latest_t - np.arange(h - 1, -1, -1, dtype=np.float64) / output_frequency_hz
        raw = {sid: [] for sid in all_ids}

        for frame in self._frames:
            frame_t = float(frame.timestamp)
            age_s = max(0.0, latest_t - frame_t)
            ego_x, ego_y, ego_yaw = self._past_ego_pose(
                age_s, ego_speed_mps, ego_yaw_rate_rps
            )
            raw["ego"].append((frame_t, ego_x, ego_y))

            c, s = np.cos(ego_yaw), np.sin(ego_yaw)
            for obj in frame.objects:
                sid = str(obj.id)
                if sid not in raw:
                    continue
                local_x, local_y = float(obj.position[0]), float(obj.position[1])
                fixed_x = ego_x + c * local_x - s * local_y
                fixed_y = ego_y + s * local_x + c * local_y
                raw[sid].append((frame_t, fixed_x, fixed_y))

        position = np.zeros((n, h, 2), dtype=np.float32)
        velocity = np.zeros((n, h, 2), dtype=np.float32)
        heading = np.zeros((n, h), dtype=np.float32)
        valid_mask = np.zeros((n, h), dtype=bool)
        agent_type = np.zeros((n,), dtype=np.int64)

        agent_type[0] = 0

        for idx, sid in enumerate(all_ids):
            samples = sorted(raw[sid], key=lambda item: item[0])
            if len(samples) < 2:
                continue
            times = np.asarray([sample[0] for sample in samples], dtype=np.float64)
            xs = np.asarray([sample[1] for sample in samples], dtype=np.float64)
            ys = np.asarray([sample[2] for sample in samples], dtype=np.float64)
            gaps = np.diff(times)
            gaps = gaps[gaps > 1e-6]
            typical_gap = float(np.median(gaps)) if gaps.size else 0.1
            max_gap = max(0.25, typical_gap * 2.5)

            for step, target_t in enumerate(grid):
                if target_t < times[0] - 1e-6 or target_t > times[-1] + 1e-6:
                    continue
                right = int(np.searchsorted(times, target_t, side="left"))
                if right < len(times) and abs(times[right] - target_t) <= 1e-6:
                    position[idx, step] = (xs[right], ys[right])
                    valid_mask[idx, step] = True
                    continue
                if right == 0 or right >= len(times):
                    continue
                left = right - 1
                gap = times[right] - times[left]
                if gap <= 0 or gap > max_gap:
                    continue
                alpha = (target_t - times[left]) / gap
                position[idx, step, 0] = xs[left] + alpha * (xs[right] - xs[left])
                position[idx, step, 1] = ys[left] + alpha * (ys[right] - ys[left])
                valid_mask[idx, step] = True

            self._differentiate_valid_segments(
                position[idx], valid_mask[idx], velocity[idx], output_frequency_hz
            )
            moving = np.linalg.norm(velocity[idx], axis=1) > 0.05
            heading[idx, moving] = np.arctan2(
                velocity[idx, moving, 1], velocity[idx, moving, 0]
            )
            if idx > 0:
                agent_type[idx] = self._type_to_int(latest_type.get(sid, "unknown"))

        keep = [0]
        for idx in range(1, n):
            if valid_mask[idx].sum() >= 2 and valid_mask[idx, -1]:
                keep.append(idx)
        if len(keep) < 2 or valid_mask[0].sum() < 2:
            return None

        keep_array = np.asarray(keep, dtype=np.int64)

        return {
            "position": position[keep_array],
            "velocity": velocity[keep_array],
            "heading": heading[keep_array],
            "valid_mask": valid_mask[keep_array],
            "agent_type": agent_type[keep_array],
            "av_index": 0,
            "coordinate_frame": "ego",
            "frequency_hz": float(output_frequency_hz),
            "agent_ids": [all_ids[index] for index in keep],
        }

    @staticmethod
    def _past_ego_pose(age_s: float, speed_mps: float, yaw_rate_rps: float):
        """过去自车位姿在最新 t0 ego 坐标系中的表达。"""
        t = -float(age_s)
        speed = float(speed_mps)
        yaw_rate = float(yaw_rate_rps)
        yaw = yaw_rate * t
        if abs(yaw_rate) < 1e-8:
            return speed * t, 0.0, 0.0
        radius = speed / yaw_rate
        return (
            float(radius * np.sin(yaw)),
            float(radius * (1.0 - np.cos(yaw))),
            float(yaw),
        )

    @staticmethod
    def _differentiate_valid_segments(position, valid, output, frequency_hz):
        """只在连续有效区间内对位置求差分速度。"""
        indices = np.flatnonzero(valid)
        if indices.size < 2:
            return
        segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        dt = 1.0 / float(frequency_hz)
        for segment in segments:
            if segment.size >= 2:
                output[segment] = np.gradient(position[segment], dt, axis=0)

    def get_latest_positions(self) -> Dict:
        """返回最新一帧各目标的位置 {id: (x, y)}。"""
        if not self._frames:
            return {}
        latest = self._frames[-1]
        result = {"ego": (0.0, 0.0)}
        for obj in latest.objects:
            result[obj.id] = tuple(obj.position)
        return result
