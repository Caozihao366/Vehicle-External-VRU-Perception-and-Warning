"""
传感器质量检测系统 - 数据类型定义
所有模块共用的枚举、数据类、结构体定义
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


# ============================================================
# 退化类型枚举
# ============================================================

class DegradationType(str, Enum):
    """图像退化类型"""
    NONE = "none"                     # 正常，无退化
    OVEREXPOSURE = "overexposure"     # 过曝
    LOW_LIGHT = "low_light"           # 低光/暗部
    RAIN_DROP = "rain_drop"           # 雨滴/水珠
    MUD = "mud"                       # 泥污/污渍
    MOTION_BLUR = "motion_blur"       # 运动模糊
    UNKNOWN = "unknown"               # 未知退化


class DegradationSeverity(str, Enum):
    """退化严重程度"""
    NONE = "none"         # 无
    MILD = "mild"         # 轻度
    MODERATE = "moderate" # 中度
    SEVERE = "severe"     # 重度


class GlobalQualityStatus(str, Enum):
    """全局图像质量状态"""
    NORMAL = "normal"                     # 正常
    PARTIALLY_DEGRADED = "partial"        # 局部退化
    DEGRADED = "degraded"                 # 整体退化
    UNAVAILABLE = "unavailable"           # 完全不可用


# ============================================================
# 区域退化信息
# ============================================================

@dataclass
class RegionDegradationInfo:
    """单个网格区域的退化信息"""
    # 网格位置
    grid_row: int = 0
    grid_col: int = 0

    # 像素坐标 bbox (x1, y1, x2, y2)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # 退化类型与严重程度
    degradation_type: DegradationType = DegradationType.NONE
    severity: DegradationSeverity = DegradationSeverity.NONE

    # 置信度（0~1）
    confidence: float = 0.0

    # 退化像素占该区域比例（0~1）
    pixel_ratio: float = 0.0

    # 该区域质量评分（0~1，1=完全正常）
    quality_score: float = 1.0

    # 原始统计特征（调试用）
    stats: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 全局质量检测结果（第一级输出）
# ============================================================

@dataclass
class GlobalQualityResult:
    """全局图像质量检测结果"""
    # 综合质量评分（0~1）
    quality_score: float = 1.0

    # 全局状态
    status: GlobalQualityStatus = GlobalQualityStatus.NORMAL

    # 疑似退化类型列表（可能有多种）
    suspected_types: List[DegradationType] = field(default_factory=list)

    # 是否触发进一步检测
    is_degraded: bool = False

    # 各维度原始统计值
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    overexposed_ratio: float = 0.0
    dark_ratio: float = 0.0
    laplacian_variance: float = 0.0
    gradient_median: float = 0.0
    edge_density: float = 0.0
    image_entropy: float = 0.0
    bright_spot_density: float = 0.0
    bright_spot_count: int = 0


# ============================================================
# 分区诊断结果（第二级输出）
# ============================================================

@dataclass
class RegionalDiagnosisResult:
    """分区精细诊断结果"""
    # 网格尺寸
    grid_rows: int = 6
    grid_cols: int = 8

    # 所有网格区域的退化信息
    regions: List[RegionDegradationInfo] = field(default_factory=list)

    # 退化掩码（像素级，可选，用于可视化）
    degradation_mask: Optional[Any] = None  # np.ndarray

    # 总退化面积占比（0~1）
    total_degraded_ratio: float = 0.0


# ============================================================
# 盲区信息（第三级输出的基本单元）
# ============================================================

@dataclass
class BlindZoneInfo:
    """盲区信息"""
    # 退化类型
    degradation_type: DegradationType = DegradationType.NONE

    # 严重程度
    severity: DegradationSeverity = DegradationSeverity.NONE

    # 盲区 bbox (x1, y1, x2, y2) 像素坐标
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # 退化像素占比（该区域内的退化比例）
    pixel_ratio: float = 0.0

    # 置信度（0~1）
    confidence: float = 0.0


# ============================================================
# 最终输出（第三级输出，供下游模块消费）
# ============================================================

@dataclass
class SensorQualityOutput:
    """
    传感器质量检测最终输出
    下游检测、跟踪、风险评估模块直接消费此结构
    """
    # 时间戳与帧ID
    timestamp: float = 0.0
    frame_id: int = 0

    # 全局状态
    global_status: GlobalQualityStatus = GlobalQualityStatus.NORMAL
    global_quality_score: float = 1.0

    # 有效视野比例（0~1，被遮挡的部分不计入有效视野）
    effective_fov_ratio: float = 1.0

    # 盲区列表（所有检测到的盲区）
    blind_zones: List[BlindZoneInfo] = field(default_factory=list)

    # 检测置信度折扣系数（0~1，下游检测模块直接乘以此系数）
    detection_confidence_discount: float = 1.0

    # 跟踪可靠性系数（0~1，跟踪模块根据此值调整外推依赖程度）
    tracking_reliability: float = 1.0

    # 系统行为建议（字符串标签，供决策层参考）
    suggestion: str = "normal_operation"

    # 原始中间结果（调试用，生产环境可忽略）
    global_result: Optional[GlobalQualityResult] = None
    regional_result: Optional[RegionalDiagnosisResult] = None
