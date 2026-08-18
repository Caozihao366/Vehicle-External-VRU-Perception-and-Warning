"""
单目地面平面投影
================
单路前视视频模式下，将 2D 像素坐标投影到 ego 坐标系（米制）。
使用简单针孔相机 + 地面平面假设，无需标定文件。

ego 坐标系: x 向前, y 向左 (与 bev_types.py / TTC / QCNet 一致)
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from bev_types import BEVFrame, EgoState, ObjectState


# COCO 类别 → bev_types type 字符串
_CLS_TO_TYPE = {
    0: "pedestrian",
    1: "cyclist",
    2: "vehicle",
    3: "motorcyclist",
    5: "bus",
    7: "vehicle",
}


@dataclass
class ProjectedObject:
    """投影后的米制目标，附带原始 track 引用。"""
    track_id: int
    cls: int
    obj_type: str
    position: Tuple[float, float]   # (x_forward, y_left) in meters
    velocity: Tuple[float, float]   # (vx, vy) in m/s
    heading: float                  # radians
    bbox: tuple
    behavior: str
    behavior_conf: float
    confidence: float = 1.0
    track: dict = None              # 原始 track dict 引用


class GroundPlaneProjector:
    """
    针孔相机地面平面投影。

    Args:
        image_size: (width, height)
        camera_height_m: 相机离地高度
        camera_pitch_deg: 相机俯角（正值=向下看）
        hfov_deg: 水平视场角
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        camera_height_m: float = 1.5,
        camera_pitch_deg: float = 5.0,
        hfov_deg: float = 70.0,
        max_dist_m: float = 50.0,
        min_dist_m: float = 1.0,
    ):
        self.w, self.h = image_size
        self.h_m = camera_height_m
        self.pitch_rad = math.radians(camera_pitch_deg)
        self.max_dist = max_dist_m
        self.min_dist = min_dist_m

        # 焦距（像素），由水平视场角推算
        self.fx = (self.w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.fy = self.fx  # 假设方像素
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0

        # 上一帧米制位置，用于差分速度
        self._prev_positions = {}  # track_id -> (x, y)

    def pixel_to_ground(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        像素坐标 → ego 地面坐标 (x_forward, y_left)，单位米。

        原理:
            α = atan2(v - cy, fy)          像素在光轴下方的角度
            β = pitch + α                    相对水平面的俯角
            forward = h / tan(β)
            left = forward * (cx - u) / fx
        """
        alpha = math.atan2(v - self.cy, self.fy)
        beta = self.pitch_rad + alpha
        if beta <= 0.01:
            return None  # 在地平线以上，无法投影到地面

        forward = self.h_m / math.tan(beta)
        if forward < self.min_dist or forward > self.max_dist:
            return None

        left = forward * (self.cx - u) / self.fx
        return (forward, left)

    def project_tracks(
        self,
        tracks: list,
        dt: float,
        ego_speed_mps: float = 0.0,
        ego_yaw_rate_rps: float = 0.0,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> Tuple[BEVFrame, list]:
        """
        将 2D tracks 投影为米制 BEVFrame。

        Args:
            tracks: vru_perception 输出的 track dict 列表
            dt: 帧间隔 (秒)
            ego_speed_mps: 自车纵向速度
            ego_yaw_rate_rps: 自车横摆角速度
            frame_id: 帧号
            timestamp: 时间戳 (秒)

        Returns:
            (BEVFrame, list[ProjectedObject])
        """
        objects_state = []
        projected = []
        current_ids = set()

        for obj in tracks:
            tid = obj["track_id"]
            current_ids.add(tid)

            # 用检测框底部中心点投影
            bx, by = obj.get("bottom_center", (None, None))
            if bx is None:
                x1, y1, x2, y2 = obj["bbox"]
                bx, by = (x1 + x2) / 2, y2

            pos = self.pixel_to_ground(bx, by)
            if pos is None:
                continue

            # 速度 = 米制位置差分
            vx, vy = 0.0, 0.0
            prev = self._prev_positions.get(tid)
            if prev is not None and dt > 0:
                vx = (pos[0] - prev[0]) / dt
                vy = (pos[1] - prev[1]) / dt
                # 限速滤波（消除投影抖动）
                speed = math.hypot(vx, vy)
                if speed > 30.0:
                    vx *= 30.0 / speed
                    vy *= 30.0 / speed

            self._prev_positions[tid] = pos

            heading = math.atan2(vy, vx) if (abs(vx) > 0.05 or abs(vy) > 0.05) else 0.0
            obj_type = _CLS_TO_TYPE.get(obj.get("cls", -1), "unknown")

            behavior = obj.get("behavior", "normal")
            behavior_conf = obj.get("behavior_conf", 1.0)

            # 构建组员接口 ObjectState
            os = ObjectState(
                id=tid,
                type=obj_type,
                position=[round(pos[0], 3), round(pos[1], 3)],
                velocity=[round(vx, 3), round(vy, 3)],
                heading=round(heading, 4),
            )
            objects_state.append(os)

            projected.append(ProjectedObject(
                track_id=tid,
                cls=obj.get("cls", -1),
                obj_type=obj_type,
                position=pos,
                velocity=(vx, vy),
                heading=heading,
                bbox=obj.get("bbox"),
                behavior=behavior,
                behavior_conf=behavior_conf,
                track=obj,
            ))

        # 清理消失目标的历史
        for tid in list(self._prev_positions.keys()):
            if tid not in current_ids:
                self._prev_positions.pop(tid, None)

        # 自车状态
        ego = EgoState(
            id="ego",
            position=[0.0, 0.0],
            velocity=[ego_speed_mps, 0.0],
            heading=0.0,
        )

        bev_frame = BEVFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            coordinate_frame="ego",
            ego=ego,
            objects=objects_state,
        )

        return bev_frame, projected
