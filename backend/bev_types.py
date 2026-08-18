
#请按照 bev_types.py 定义的接口输出实时 BEVFrame 数据。每10Hz输出当前帧目标状态，包括自车状态和所有目标状态。
# 目标需要经过MOT跟踪，保证id跨帧一致，所有位置速度航向均为ego坐标系。
from dataclasses import dataclass
from typing import List, Union


@dataclass
class EgoState:
    """自车当前状态"""

    # 自车ID
    id: str

    # 当前自车位置
    # 单位：m
    # ego坐标系下固定为 [0, 0]
    position: List[float]       # [x, y]

    # 当前自车速度
    # 单位：m/s
    velocity: List[float]       # [vx, vy]

    # 当前自车航向
    # 单位：rad
    # ego坐标系下固定为 0
    heading: float



@dataclass
class ObjectState:
    """目标当前状态"""

    # MOT稳定Track ID
    # 同一个物体跨帧必须保持一致
    id: Union[str, int]

    # 目标类别
    # vehicle / pedestrian / cyclist /
    # motorcyclist / bus / unknown
    type: str

    # 当前目标位置
    # ego坐标系
    # 单位：m
    position: List[float]       # [x, y]

    # 当前目标速度
    # ego坐标系
    # 单位：m/s
    velocity: List[float]       # [vx, vy]

    # 当前目标航向
    # 单位：rad
    heading: float



@dataclass
class BEVFrame:
    """BEV实时输出的一帧数据"""

    # 当前帧编号
    frame_id: int

    # 当前帧时间戳
    # 单位：秒
    timestamp: float

    # 坐标系
    # 固定填写 "ego"
    coordinate_frame: str

    # 当前自车状态
    ego: EgoState

    # 当前帧检测/跟踪目标列表
    objects: List[ObjectState]