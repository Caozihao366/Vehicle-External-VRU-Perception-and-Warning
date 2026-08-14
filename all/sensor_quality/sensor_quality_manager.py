"""
传感器质量检测系统 - 主管理器
串联三级检测流程，提供统一的对外接口。

处理流程：
1. 第一级：全局快速体检（GlobalQualityChecker）
   - 正常帧直接放行，不进入后续计算
   - 异常帧触发第二级
2. 第二级：分区精细诊断（RegionalDiagnoser）
   - 网格化定位退化区域
   - 识别退化类型与严重程度
3. 第三级：降级决策（DegradationDecisionMaker）
   - 时序平滑防抖
   - 盲区聚合
   - 输出标准化降级结果
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from .data_types import (
    SensorQualityOutput,
    GlobalQualityStatus,
    DegradationType,
)
from .global_quality_checker import GlobalQualityChecker, GlobalCheckerConfig
from .regional_diagnoser import RegionalDiagnoser, RegionalDiagnoserConfig
from .degradation_decision import DegradationDecisionMaker, DegradationDecisionConfig
from . import utils


# ============================================================
# 主配置
# ============================================================

@dataclass
class SensorQualityConfig:
    """传感器质量检测系统总配置"""
    # 是否启用第二级（关闭则只做全局检测，更快）
    enable_regional_diagnosis: bool = True

    # 是否启用第三级时序平滑（关闭则直接输出当前帧结果）
    enable_temporal_smoothing: bool = True

    # 各子模块配置
    global_config: GlobalCheckerConfig = None
    regional_config: RegionalDiagnoserConfig = None
    decision_config: DegradationDecisionConfig = None

    # 性能统计
    enable_profiling: bool = False


# ============================================================
# 主管理器
# ============================================================

class SensorQualityManager:
    """
    传感器质量检测系统主管理器
    对外统一接口，串联三级检测流程。

    典型用法：
        manager = SensorQualityManager()
        result = manager.process_frame(image, timestamp=ts, frame_id=100)
        print(result.global_status, result.blind_zones)
    """

    def __init__(self, config: SensorQualityConfig = None):
        self.config = config or SensorQualityConfig()

        # 初始化各子模块
        self.global_checker = GlobalQualityChecker(self.config.global_config)
        self.regional_diagnoser = RegionalDiagnoser(self.config.regional_config)
        self.decision_maker = DegradationDecisionMaker(self.config.decision_config)

        # 帧计数
        self._frame_count = 0

        # 性能统计
        self._profiling = {
            'global_time': [],
            'regional_time': [],
            'decision_time': [],
            'total_time': [],
        }

    def process_frame(self,
                      image: np.ndarray,
                      timestamp: float = None,
                      frame_id: int = None) -> SensorQualityOutput:
        """
        处理单帧图像，输出完整的传感器质量检测结果。

        Args:
            image: 输入图像 (BGR格式，numpy数组)
            timestamp: 时间戳（秒），None则使用当前时间
            frame_id: 帧ID，None则自动递增

        Returns:
            SensorQualityOutput 完整检测结果
        """
        self._frame_count += 1
        if timestamp is None:
            timestamp = time.time()
        if frame_id is None:
            frame_id = self._frame_count

        t_total_start = time.time()

        # ============================================================
        # 第一级：全局快速体检
        # ============================================================
        t0 = time.time()
        global_result = self.global_checker.check(image)
        t_global = time.time() - t0

        # 正常帧快速放行：不触发第二级，直接输出正常结果
        if (not global_result.is_degraded
                and global_result.status == GlobalQualityStatus.NORMAL
                and not self.config.enable_regional_diagnosis):
            t_total = time.time() - t_total_start
            self._record_profiling(t_global, 0, 0, t_total)

            # 更新基线
            self.global_checker.update_baseline(global_result)

            return SensorQualityOutput(
                timestamp=timestamp,
                frame_id=frame_id,
                global_status=GlobalQualityStatus.NORMAL,
                global_quality_score=global_result.quality_score,
                effective_fov_ratio=1.0,
                blind_zones=[],
                detection_confidence_discount=1.0,
                tracking_reliability=1.0,
                suggestion="normal_operation",
                global_result=global_result,
                regional_result=None,
            )

        # ============================================================
        # 第二级：分区精细诊断
        # ============================================================
        if self.config.enable_regional_diagnosis:
            t0 = time.time()
            regional_result = self.regional_diagnoser.diagnose(
                image, global_result
            )
            t_regional = time.time() - t0
        else:
            regional_result = None
            t_regional = 0

        # ============================================================
        # 第三级：降级决策
        # ============================================================
        if self.config.enable_temporal_smoothing and regional_result:
            t0 = time.time()
            output = self.decision_maker.decide(
                regional_result, global_result,
                timestamp=timestamp, frame_id=frame_id,
            )
            t_decision = time.time() - t0
        elif regional_result:
            # 无时序平滑，直接组装结果
            t_decision = 0
            output = self._assemble_direct_output(
                image, global_result, regional_result, timestamp, frame_id
            )
        else:
            t_decision = 0
            output = SensorQualityOutput(
                timestamp=timestamp,
                frame_id=frame_id,
                global_status=global_result.status,
                global_quality_score=global_result.quality_score,
                effective_fov_ratio=1.0,
                blind_zones=[],
                detection_confidence_discount=1.0,
                tracking_reliability=1.0,
                suggestion="normal_operation" if global_result.status == GlobalQualityStatus.NORMAL else "check_regional",
                global_result=global_result,
                regional_result=None,
            )

        # ============================================================
        # 收尾
        # ============================================================
        t_total = time.time() - t_total_start
        self._record_profiling(t_global, t_regional, t_decision, t_total)

        # 更新基线（正常帧更新熵基线等）
        if global_result.status == GlobalQualityStatus.NORMAL:
            self.global_checker.update_baseline(global_result)

        return output

    # ============================================================
    # 直接组装输出（无时序平滑时使用）
    # ============================================================

    def _assemble_direct_output(self,
                                 image: np.ndarray,
                                 global_result,
                                 regional_result,
                                 timestamp: float,
                                 frame_id: int) -> SensorQualityOutput:
        """无时序平滑时，直接从分区结果组装输出"""
        from .data_types import BlindZoneInfo

        blind_zones = []
        for region in regional_result.regions:
            if region.degradation_type != DegradationType.NONE:
                blind_zones.append(BlindZoneInfo(
                    degradation_type=region.degradation_type,
                    severity=region.severity,
                    bbox=region.bbox,
                    pixel_ratio=region.pixel_ratio,
                    confidence=region.confidence,
                ))

        # 简单计算有效视野比例
        total_area = sum(
            (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])
            for r in regional_result.regions
        )
        degraded_area = sum(
            (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]) * r.pixel_ratio
            for r in regional_result.regions
            if r.degradation_type != DegradationType.NONE
        )
        effective_fov = 1.0 - (degraded_area / total_area) if total_area > 0 else 1.0

        # 降级等级
        degraded_ratio = degraded_area / total_area if total_area > 0 else 0
        if degraded_ratio > 0.3:
            status = GlobalQualityStatus.DEGRADED
            discount = 0.3
            reliability = 0.4
        elif degraded_ratio > 0.1:
            status = GlobalQualityStatus.PARTIALLY_DEGRADED
            discount = 0.6
            reliability = 0.7
        elif degraded_ratio > 0.02:
            status = GlobalQualityStatus.PARTIALLY_DEGRADED
            discount = 0.9
            reliability = 0.95
        else:
            status = GlobalQualityStatus.NORMAL
            discount = 1.0
            reliability = 1.0

        return SensorQualityOutput(
            timestamp=timestamp,
            frame_id=frame_id,
            global_status=status,
            global_quality_score=global_result.quality_score,
            effective_fov_ratio=max(0.0, effective_fov),
            blind_zones=blind_zones,
            detection_confidence_discount=discount,
            tracking_reliability=reliability,
            suggestion=f"degraded_ratio_{degraded_ratio:.2f}",
            global_result=global_result,
            regional_result=regional_result,
        )

    # ============================================================
    # 性能统计
    # ============================================================

    def _record_profiling(self, t_global, t_regional, t_decision, t_total):
        """记录性能数据"""
        if not self.config.enable_profiling:
            return
        self._profiling['global_time'].append(t_global * 1000)  # ms
        self._profiling['regional_time'].append(t_regional * 1000)
        self._profiling['decision_time'].append(t_decision * 1000)
        self._profiling['total_time'].append(t_total * 1000)

        # 保留最近1000帧
        for key in self._profiling:
            if len(self._profiling[key]) > 1000:
                self._profiling[key] = self._profiling[key][-1000:]

    def get_profiling_stats(self) -> dict:
        """获取性能统计信息"""
        if not self.config.enable_profiling:
            return {}

        stats = {}
        for key, values in self._profiling.items():
            if values:
                stats[key] = {
                    'avg_ms': sum(values) / len(values),
                    'max_ms': max(values),
                    'min_ms': min(values),
                    'count': len(values),
                }
        return stats

    # ============================================================
    # 可视化（调试用）
    # ============================================================

    def visualize(self,
                  image: np.ndarray,
                  result: SensorQualityOutput,
                  mode: str = "blind_zones") -> np.ndarray:
        """
        可视化检测结果
        Args:
            image: 原始图像
            result: 检测结果
            mode: "blind_zones" - 只画盲区
                  "grid" - 画网格质量评分
                  "both" - 两者都画
        """
        vis = image.copy()

        if mode in ("grid", "both") and result.regional_result:
            vis = utils.draw_grid_debug(
                vis,
                result.regional_result.regions,
                result.regional_result.grid_rows,
                result.regional_result.grid_cols,
            )

        if mode in ("blind_zones", "both"):
            vis = utils.draw_blind_zones(vis, result.blind_zones)

        # 顶部状态栏
        status_text = f"Status: {result.global_status.value} | " \
                      f"Score: {result.global_quality_score:.2f} | " \
                      f"FOV: {result.effective_fov_ratio:.1%} | " \
                      f"Discount: {result.detection_confidence_discount:.1%}"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(vis, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return vis

    # ============================================================
    # 重置
    # ============================================================

    def reset(self):
        """重置所有状态（清空时序缓存等）"""
        self._frame_count = 0
        self.regional_diagnoser._history_buffer.clear()
        self.decision_maker._regional_history.clear()
        self.decision_maker._global_history.clear()
        for key in self._profiling:
            self._profiling[key].clear()
