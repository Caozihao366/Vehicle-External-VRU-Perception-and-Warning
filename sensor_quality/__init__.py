"""
传感器质量检测与降级系统 (Sensor Quality Detection & Degradation System)

FR-08: 传感器质量检测与降级
- 识别暴雨水滴、泥污、过曝/低光导致的不可见区域
- 输出视野盲区标记
- 局部遮挡不导致全局失效，失效区域明确上报

三级架构：
1. GlobalQualityChecker  - 全局快速体检（统计特征粗筛）
2. RegionalDiagnoser     - 分区精细诊断（网格化定位）
3. DegradationDecisionMaker - 降级决策（时序平滑 + 盲区聚合 + 分级输出）
"""

from .data_types import (
    DegradationType,
    DegradationSeverity,
    GlobalQualityStatus,
    RegionDegradationInfo,
    GlobalQualityResult,
    RegionalDiagnosisResult,
    BlindZoneInfo,
    SensorQualityOutput,
)
from .global_quality_checker import GlobalQualityChecker, GlobalCheckerConfig
from .regional_diagnoser import RegionalDiagnoser, RegionalDiagnoserConfig
from .degradation_decision import DegradationDecisionMaker, DegradationDecisionConfig
from .sensor_quality_manager import SensorQualityManager, SensorQualityConfig

__all__ = [
    # 类型
    'DegradationType',
    'DegradationSeverity',
    'GlobalQualityStatus',
    'RegionDegradationInfo',
    'GlobalQualityResult',
    'RegionalDiagnosisResult',
    'BlindZoneInfo',
    'SensorQualityOutput',
    # 配置
    'GlobalCheckerConfig',
    'RegionalDiagnoserConfig',
    'DegradationDecisionConfig',
    'SensorQualityConfig',
    # 核心类
    'GlobalQualityChecker',
    'RegionalDiagnoser',
    'DegradationDecisionMaker',
    'SensorQualityManager',
]
