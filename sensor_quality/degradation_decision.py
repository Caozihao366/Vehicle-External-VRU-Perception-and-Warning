"""
传感器质量检测系统 - 第三级：降级决策与盲区标记输出
接收第二级的分区诊断结果，进行时序平滑后，
输出结构化的盲区标记与系统降级建议。

核心职责：
- 时序平滑：滑动窗口投票，防止单帧误判
- 盲区聚合：将相邻退化网格合并为连续盲区区域
- 降级分级：轻度/中度/重度，对应不同的系统行为
- 输出标准化：生成下游可直接消费的盲区信息
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

from .data_types import (
    DegradationType,
    DegradationSeverity,
    GlobalQualityStatus,
    RegionDegradationInfo,
    RegionalDiagnosisResult,
    GlobalQualityResult,
    BlindZoneInfo,
    SensorQualityOutput,
)


# ============================================================
# 配置参数
# ============================================================

@dataclass
class DegradationDecisionConfig:
    """降级决策配置"""
    # 时序平滑
    temporal_window_size: int = 3        # 滑动窗口大小（帧数）
    temporal_vote_threshold: float = 0.6  # 投票通过比例（如3帧中2帧）

    # 盲区合并
    merge_adjacent: bool = True           # 是否合并相邻退化区域
    merge_same_type_only: bool = True     # 只合并同类型退化
    merge_max_gap: int = 1                # 合并时允许的最大网格间隔

    # 降级分级阈值（按总退化面积占比）
    mild_degradation_ratio: float = 0.10   # 轻度：< 10%
    moderate_degradation_ratio: float = 0.30  # 中度：10% ~ 30%
    severe_degradation_ratio: float = 0.50   # 重度：> 30%

    # 置信度折扣
    confidence_discount_mild: float = 0.9    # 轻度：90%
    confidence_discount_moderate: float = 0.6  # 中度：60%
    confidence_discount_severe: float = 0.3    # 重度：30%

    # 跟踪可靠性
    tracking_reliability_mild: float = 0.95
    tracking_reliability_moderate: float = 0.7
    tracking_reliability_severe: float = 0.4

    # 最小置信度（低于此值的退化不输出）
    min_blind_zone_confidence: float = 0.4


# ============================================================
# 降级决策器
# ============================================================

class DegradationDecisionMaker:
    """
    第三级：降级决策与盲区标记输出
    输入：分区诊断结果 + 全局检测结果
    输出：SensorQualityOutput（最终结果，含盲区列表与降级建议）
    """

    def __init__(self, config: DegradationDecisionConfig = None):
        self.config = config or DegradationDecisionConfig()

        # 时序缓存
        self._regional_history = deque(maxlen=10)
        self._global_history = deque(maxlen=10)
        self._frame_count = 0

    def decide(self,
               regional_result: RegionalDiagnosisResult,
               global_result: GlobalQualityResult = None,
               timestamp: float = 0.0,
               frame_id: int = 0) -> SensorQualityOutput:
        """
        执行降级决策
        Args:
            regional_result: 第二级分区诊断结果
            global_result: 第一级全局检测结果
            timestamp: 时间戳
            frame_id: 帧ID
        Returns:
            SensorQualityOutput
        """
        self._frame_count += 1

        # 1. 时序平滑
        smoothed_regions = self._temporal_smoothing(regional_result.regions)

        # 2. 盲区聚合（合并相邻退化网格）
        blind_zones = self._merge_into_blind_zones(
            smoothed_regions,
            regional_result.grid_rows,
            regional_result.grid_cols,
        )

        # 3. 计算总退化面积占比
        total_degraded_ratio = self._calc_total_degraded_ratio(smoothed_regions)

        # 4. 判定全局降级等级
        global_status, severity_level = self._determine_global_status(
            total_degraded_ratio, global_result
        )

        # 5. 计算置信度折扣与跟踪可靠性
        confidence_discount = self._calc_confidence_discount(severity_level)
        tracking_reliability = self._calc_tracking_reliability(severity_level)

        # 6. 有效视野比例
        effective_fov_ratio = max(0.0, 1.0 - total_degraded_ratio)

        # 7. 系统行为建议
        suggestion = self._generate_suggestion(
            global_status, severity_level, blind_zones
        )

        # 8. 组装最终输出
        output = SensorQualityOutput(
            timestamp=timestamp,
            frame_id=frame_id,
            global_status=global_status,
            global_quality_score=global_result.quality_score if global_result else 1.0,
            effective_fov_ratio=effective_fov_ratio,
            blind_zones=blind_zones,
            detection_confidence_discount=confidence_discount,
            tracking_reliability=tracking_reliability,
            suggestion=suggestion,
            global_result=global_result,
            regional_result=regional_result,
        )

        # 9. 更新历史
        self._regional_history.append(smoothed_regions)
        if global_result:
            self._global_history.append(global_result)

        return output

    # ============================================================
    # 时序平滑
    # ============================================================

    def _temporal_smoothing(self,
                            current_regions: List[RegionDegradationInfo]
                            ) -> List[RegionDegradationInfo]:
        """
        时序平滑：滑动窗口投票
        只有连续多帧都检测到退化，才确认该区域退化。
        防止单帧误判（如雨刷经过、瞬时闪光等）。
        """
        window_size = self.config.temporal_window_size
        vote_threshold = self.config.temporal_vote_threshold

        # 历史帧数不足时，直接返回当前结果（但降低置信度）
        if len(self._regional_history) < window_size - 1:
            # 前几帧降低置信度
            result = []
            for region in current_regions:
                smoothed = RegionDegradationInfo(
                    grid_row=region.grid_row,
                    grid_col=region.grid_col,
                    bbox=region.bbox,
                    degradation_type=region.degradation_type,
                    severity=region.severity,
                    confidence=region.confidence * 0.7,  # 初始帧置信度打折
                    pixel_ratio=region.pixel_ratio,
                    quality_score=region.quality_score,
                    stats=region.stats,
                )
                result.append(smoothed)
            return result

        # 取最近 N-1 帧历史 + 当前帧
        history = list(self._regional_history)[-(window_size - 1):]
        all_frames = history + [current_regions]

        smoothed_regions = []
        num_frames = len(all_frames)

        for i, current_region in enumerate(current_regions):
            # 严重程度数值映射
            severity_values = {
                DegradationSeverity.NONE: 0,
                DegradationSeverity.MILD: 1,
                DegradationSeverity.MODERATE: 2,
                DegradationSeverity.SEVERE: 3,
            }

            # 统计该网格在历史帧中的退化情况
            type_votes = {}
            severity_sum = {}
            confidence_sum = {}
            pixel_ratio_sum = 0.0
            degraded_frames = 0

            for frame in all_frames:
                if i >= len(frame):
                    continue
                hist_region = frame[i]
                if hist_region.degradation_type != DegradationType.NONE:
                    degraded_frames += 1
                    dtype = hist_region.degradation_type
                    type_votes[dtype] = type_votes.get(dtype, 0) + 1
                    severity_sum[dtype] = severity_sum.get(dtype, 0) + severity_values[hist_region.severity]
                    confidence_sum[dtype] = confidence_sum.get(dtype, 0) + hist_region.confidence
                    pixel_ratio_sum += hist_region.pixel_ratio

            # 投票决定是否退化
            degraded_ratio = degraded_frames / num_frames

            if degraded_ratio >= vote_threshold and type_votes:
                # 取票数最多的类型
                best_type = max(type_votes, key=type_votes.get)
                avg_confidence = confidence_sum[best_type] / type_votes[best_type]
                avg_pixel_ratio = pixel_ratio_sum / degraded_frames

                # 严重程度取平均
                avg_severity_val = severity_sum[best_type] / type_votes[best_type]
                # 映射回枚举
                if avg_severity_val >= 2.5:
                    best_severity = DegradationSeverity.SEVERE
                elif avg_severity_val >= 1.5:
                    best_severity = DegradationSeverity.MODERATE
                else:
                    best_severity = DegradationSeverity.MILD

                smoothed = RegionDegradationInfo(
                    grid_row=current_region.grid_row,
                    grid_col=current_region.grid_col,
                    bbox=current_region.bbox,
                    degradation_type=best_type,
                    severity=best_severity,
                    confidence=min(avg_confidence * degraded_ratio, 1.0),
                    pixel_ratio=avg_pixel_ratio,
                    quality_score=max(0.0, 1.0 - avg_confidence * degraded_ratio),
                    stats=current_region.stats,
                )
            else:
                # 未通过投票，判定为正常
                smoothed = RegionDegradationInfo(
                    grid_row=current_region.grid_row,
                    grid_col=current_region.grid_col,
                    bbox=current_region.bbox,
                    degradation_type=DegradationType.NONE,
                    severity=DegradationSeverity.NONE,
                    confidence=0.0,
                    pixel_ratio=0.0,
                    quality_score=1.0,
                    stats=current_region.stats,
                )

            smoothed_regions.append(smoothed)

        return smoothed_regions

    # ============================================================
    # 盲区聚合
    # ============================================================

    def _merge_into_blind_zones(self,
                                regions: List[RegionDegradationInfo],
                                grid_rows: int,
                                grid_cols: int) -> List[BlindZoneInfo]:
        """
        将相邻的退化网格合并为连续的盲区区域
        使用连通域分析（基于网格邻接关系）
        """
        # 过滤掉低置信度的区域
        valid_regions = [
            r for r in regions
            if r.degradation_type != DegradationType.NONE
            and r.confidence >= self.config.min_blind_zone_confidence
        ]

        if not valid_regions:
            return []

        if not self.config.merge_adjacent:
            # 不合并，每个网格单独作为一个盲区
            return [self._region_to_blind_zone(r) for r in valid_regions]

        # 构建网格索引映射
        grid_map = {}
        for r in regions:
            grid_map[(r.grid_row, r.grid_col)] = r

        # 连通域标记（BFS）
        visited = set()
        blind_zones = []

        for region in valid_regions:
            key = (region.grid_row, region.grid_col)
            if key in visited:
                continue

            # BFS 找连通域
            component = []
            queue = [key]
            visited.add(key)
            component_type = region.degradation_type

            while queue:
                curr_key = queue.pop(0)
                curr_region = grid_map.get(curr_key)
                if curr_region is None:
                    continue

                # 检查是否可以加入当前连通域
                if curr_region.degradation_type == DegradationType.NONE:
                    continue
                if self.config.merge_same_type_only and curr_region.degradation_type != component_type:
                    continue
                if curr_region.confidence < self.config.min_blind_zone_confidence:
                    continue

                component.append(curr_region)

                # 检查四邻域
                row, col = curr_key
                neighbors = [
                    (row - 1, col), (row + 1, col),
                    (row, col - 1), (row, col + 1),
                ]
                for n_key in neighbors:
                    if n_key not in visited and n_key in grid_map:
                        n_region = grid_map[n_key]
                        if (n_region.degradation_type != DegradationType.NONE
                            and n_region.confidence >= self.config.min_blind_zone_confidence):
                            if (not self.config.merge_same_type_only
                                    or n_region.degradation_type == component_type):
                                visited.add(n_key)
                                queue.append(n_key)

            if component:
                blind_zone = self._merge_component(component)
                blind_zones.append(blind_zone)

        return blind_zones

    def _merge_component(self, component: List[RegionDegradationInfo]) -> BlindZoneInfo:
        """将一个连通域（多个网格）合并为一个盲区"""
        # 计算合并后的 bbox
        x1 = min(r.bbox[0] for r in component)
        y1 = min(r.bbox[1] for r in component)
        x2 = max(r.bbox[2] for r in component)
        y2 = max(r.bbox[3] for r in component)

        # 取最常见的退化类型
        type_counts = {}
        for r in component:
            type_counts[r.degradation_type] = type_counts.get(r.degradation_type, 0) + 1
        best_type = max(type_counts, key=type_counts.get)

        # 平均置信度
        avg_confidence = sum(r.confidence for r in component) / len(component)

        # 最严重的等级
        severity_order = {
            DegradationSeverity.NONE: 0,
            DegradationSeverity.MILD: 1,
            DegradationSeverity.MODERATE: 2,
            DegradationSeverity.SEVERE: 3,
        }
        max_severity = max(component, key=lambda r: severity_order[r.severity]).severity

        # 退化像素比例（加权）
        total_area = sum((r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]) for r in component)
        weighted_pixel_ratio = sum(
            r.pixel_ratio * (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])
            for r in component
        ) / total_area if total_area > 0 else 0

        return BlindZoneInfo(
            degradation_type=best_type,
            severity=max_severity,
            bbox=(x1, y1, x2, y2),
            pixel_ratio=weighted_pixel_ratio,
            confidence=avg_confidence,
        )

    def _region_to_blind_zone(self, region: RegionDegradationInfo) -> BlindZoneInfo:
        """单个区域转为盲区信息"""
        return BlindZoneInfo(
            degradation_type=region.degradation_type,
            severity=region.severity,
            bbox=region.bbox,
            pixel_ratio=region.pixel_ratio,
            confidence=region.confidence,
        )

    # ============================================================
    # 降级等级判定
    # ============================================================

    def _calc_total_degraded_ratio(self, regions: List[RegionDegradationInfo]) -> float:
        """计算总退化面积占比"""
        if not regions:
            return 0.0

        total_area = 0
        degraded_area = 0
        for r in regions:
            cell_area = (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])
            total_area += cell_area
            if r.degradation_type != DegradationType.NONE:
                degraded_area += cell_area * r.pixel_ratio

        return degraded_area / total_area if total_area > 0 else 0.0

    def _determine_global_status(self,
                                 degraded_ratio: float,
                                 global_result: Optional[GlobalQualityResult]
                                 ) -> Tuple[GlobalQualityStatus, str]:
        """
        判定全局降级状态
        Returns:
            (全局状态, 严重程度等级: mild/moderate/severe)
        """
        if degraded_ratio >= self.config.severe_degradation_ratio:
            return GlobalQualityStatus.DEGRADED, "severe"
        elif degraded_ratio >= self.config.moderate_degradation_ratio:
            return GlobalQualityStatus.PARTIALLY_DEGRADED, "moderate"
        elif degraded_ratio >= self.config.mild_degradation_ratio:
            return GlobalQualityStatus.PARTIALLY_DEGRADED, "mild"
        else:
            # 面积很小，但如果全局质量评分低，也标记为局部退化
            if global_result and global_result.quality_score < 0.85:
                return GlobalQualityStatus.PARTIALLY_DEGRADED, "mild"
            return GlobalQualityStatus.NORMAL, "none"

    def _calc_confidence_discount(self, severity_level: str) -> float:
        """计算检测置信度折扣系数"""
        if severity_level == "severe":
            return self.config.confidence_discount_severe
        elif severity_level == "moderate":
            return self.config.confidence_discount_moderate
        elif severity_level == "mild":
            return self.config.confidence_discount_mild
        else:
            return 1.0

    def _calc_tracking_reliability(self, severity_level: str) -> float:
        """计算跟踪可靠性系数"""
        if severity_level == "severe":
            return self.config.tracking_reliability_severe
        elif severity_level == "moderate":
            return self.config.tracking_reliability_moderate
        elif severity_level == "mild":
            return self.config.tracking_reliability_mild
        else:
            return 1.0

    # ============================================================
    # 系统行为建议
    # ============================================================

    def _generate_suggestion(self,
                             status: GlobalQualityStatus,
                             severity_level: str,
                             blind_zones: List[BlindZoneInfo]) -> str:
        """生成系统行为建议"""
        if status == GlobalQualityStatus.NORMAL:
            return "normal_operation"

        # 分析主要退化类型
        type_counts = {}
        for zone in blind_zones:
            type_counts[zone.degradation_type.value] = type_counts.get(zone.degradation_type.value, 0) + 1

        dominant_type = max(type_counts, key=type_counts.get) if type_counts else "unknown"

        suggestions = {
            ("PARTIALLY_DEGRADED", "mild"): f"mild_degradation_{dominant_type}_reduce_confidence",
            ("PARTIALLY_DEGRADED", "moderate"): f"moderate_degradation_{dominant_type}_tracking_extrapolation",
            ("DEGRADED", "severe"): f"severe_degradation_{dominant_type}_sensor_unreliable",
        }

        key = (status.value, severity_level)
        return suggestions.get(key, f"degraded_{dominant_type}")
