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

    def build_qcnet_input(self) -> Optional[dict]:
        """
        构建 QCNet 输入 dict。

        Returns:
            dict with position/velocity/heading/valid_mask/agent_type/av_index
            或 None（历史不足时）
        """
        if not self.ready():
            return None

        # 收集所有出现过的 track_id（最近一帧的目标 + 自车）
        latest = self._frames[-1]
        all_ids = ["ego"]  # 自车始终在 index 0
        for obj in latest.objects:
            sid = str(obj.id)
            if sid not in all_ids:
                all_ids.append(sid)

        n = len(all_ids)
        h = self.history_steps

        position = np.zeros((n, h, 2), dtype=np.float32)
        velocity = np.zeros((n, h, 2), dtype=np.float32)
        heading = np.zeros((n, h), dtype=np.float32)
        valid_mask = np.zeros((n, h), dtype=bool)
        agent_type = np.zeros((n,), dtype=np.int64)

        # 自车类型
        agent_type[0] = 0  # vehicle

        # 填充历史
        # _frames 中最老的帧对应 step 0，最新帧对应 step h-1
        # 如果帧数不足 h，前面补 invalid
        n_frames = len(self._frames)
        start_step = h - n_frames

        for step_offset, frame in enumerate(self._frames):
            step = start_step + step_offset

            # 自车：ego 坐标系下始终 [0,0]，速度为 ego 速度
            position[0, step] = [0.0, 0.0]
            velocity[0, step] = frame.ego.velocity
            heading[0, step] = 0.0
            valid_mask[0, step] = True

            # 各目标
            for obj in frame.objects:
                sid = str(obj.id)
                if sid in all_ids:
                    idx = all_ids.index(sid)
                    position[idx, step] = obj.position
                    velocity[idx, step] = obj.velocity
                    heading[idx, step] = obj.heading
                    valid_mask[idx, step] = True
                    agent_type[idx] = self._agent_types.get(
                        obj.id, self._type_to_int(obj.type)
                    )

        # 过滤有效历史不足 2 帧的目标（QCNet 要求至少 2 个有效样本）
        keep = [0]  # 始终保留 ego
        for idx in range(1, n):
            if valid_mask[idx].sum() >= 2:
                keep.append(idx)
        keep = np.array(keep, dtype=np.int64)

        if len(keep) < 2:
            return None  # 除 ego 外没有足够历史的目标

        position = position[keep]
        velocity = velocity[keep]
        heading = heading[keep]
        valid_mask = valid_mask[keep]
        agent_type = agent_type[keep]
        all_ids = [all_ids[i] for i in keep]

        return {
            "position": position,
            "velocity": velocity,
            "heading": heading,
            "valid_mask": valid_mask,
            "agent_type": agent_type,
            "av_index": 0,
            "coordinate_frame": "ego",
            "frequency_hz": 10.0,
            "agent_ids": all_ids,
        }

    def get_latest_positions(self) -> Dict:
        """返回最新一帧各目标的位置 {id: (x, y)}。"""
        if not self._frames:
            return {}
        latest = self._frames[-1]
        result = {"ego": (0.0, 0.0)}
        for obj in latest.objects:
            result[obj.id] = tuple(obj.position)
        return result
