"""
传感器质量检测系统 - 第二级：分区精细诊断
将图像划分为网格，对每个网格独立进行退化检测，
精确定位退化区域并识别退化类型。

核心设计：
- 网格化：6×8 = 48个网格，每个独立评估
- 多类型识别：过曝、低光、雨滴、泥污、运动模糊
- 泥污 vs 雨滴区分：时序稳定性验证
- 输出：每个网格的退化类型 + 严重程度 + 置信度
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

from .data_types import (
    DegradationType,
    DegradationSeverity,
    RegionDegradationInfo,
    RegionalDiagnosisResult,
    GlobalQualityResult,
)
from . import utils


# ============================================================
# 配置参数
# ============================================================

@dataclass
class RegionalDiagnoserConfig:
    """分区诊断器配置"""
    # 网格划分（v5版：12×16）
    grid_rows: int = 12
    grid_cols: int = 16

    # 降采样尺寸
    downsample_max_size: int = 640

    # ---- 过曝阈值 ----
    overexposure_mild_ratio: float = 0.20
    overexposure_moderate_ratio: float = 0.40
    overexposure_severe_ratio: float = 0.65
    overexposure_edge_density_max: float = 0.02

    # ---- 低光阈值（v5版） ----
    lowlight_mild_ratio: float = 0.40
    lowlight_moderate_ratio: float = 0.65
    lowlight_severe_ratio: float = 0.85
    lowlight_brightness_max: float = 50.0

    # ---- 雨滴阈值（v5版：基于边缘圆形检测） ----
    rain_mild_edge_circle_density: float = 0.15
    rain_severe_edge_circle_density: float = 0.8
    rain_circularity_min: float = 0.45
    rain_edge_density_weight: float = 0.2

    # ---- 泥污阈值（v5版） ----
    mud_bottom_ratio: float = 0.25
    mud_min_area_ratio: float = 0.01
    mud_mild_area_ratio: float = 0.08
    mud_severe_area_ratio: float = 0.3
    mud_color_severe_ratio: float = 0.5
    mud_temporal_stable_frames: int = 3

    # ---- 模糊阈值 ----
    blur_mild_laplacian: float = 80.0
    blur_moderate_laplacian: float = 40.0
    blur_severe_laplacian: float = 15.0

    # ---- 通用（v5版） ----
    min_confidence: float = 0.65
    normal_entropy_baseline: float = 6.5


# ============================================================
# 分区诊断器
# ============================================================

class RegionalDiagnoser:
    """
    第二级：分区精细诊断
    输入：原始图像 + 全局检测结果（可选）
    输出：RegionalDiagnosisResult（每个网格的退化信息）
    """

    def __init__(self, config: RegionalDiagnoserConfig = None):
        self.config = config or RegionalDiagnoserConfig()
        self._baseline_entropy = self.config.normal_entropy_baseline

        # 时序缓存（用于泥污检测的时序稳定性验证）
        self._history_buffer = deque(maxlen=10)
        self._frame_count = 0

    def diagnose(self,
                 image: np.ndarray,
                 global_result: Optional[GlobalQualityResult] = None
                 ) -> RegionalDiagnosisResult:
        """
        执行分区诊断
        Args:
            image: 输入图像 (BGR格式)
            global_result: 全局检测结果（可选，用于加速）
        Returns:
            RegionalDiagnosisResult
        """
        self._frame_count += 1

        orig_h, orig_w = image.shape[:2]

        # 1. 降采样
        img_small = utils.downsample(image, self.config.downsample_max_size)
        gray = utils.to_grayscale(img_small)
        h, w = gray.shape

        # 计算缩放比例（用于把坐标缩放回原图）
        scale_x = orig_w / w
        scale_y = orig_h / h

        # 2. 划分网格
        bboxes = utils.divide_grid(
            (h, w),
            self.config.grid_rows,
            self.config.grid_cols,
        )

        # 3. 提前计算整图的泥污连通域（只算一次，效率高）
        mud_ratio_global, mud_bboxes = utils.detect_mud_connected_regions(
            img_small,
            bottom_ratio=self.config.mud_bottom_ratio,
            min_area_ratio=self.config.mud_min_area_ratio,
        )

        # 4. 逐网格诊断
        regions = []
        for idx, bbox in enumerate(bboxes):
            row = idx // self.config.grid_cols
            col = idx % self.config.grid_cols

            cell_gray = utils.extract_grid_cell(gray, bbox)
            cell_color = utils.extract_grid_cell(img_small, bbox)

            region_info = self._diagnose_single_region(
                cell_gray, cell_color, row, col, bbox, mud_bboxes
            )
            regions.append(region_info)

        # 5. 把所有bbox缩放回原图坐标（重要！修复坐标错位bug）
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            region.bbox = (
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            )

        # 6. 泥污时序稳定性验证
        regions = self._verify_mud_temporal_stability(regions)

        # 7. 保存当前帧到历史
        self._history_buffer.append(regions)

        # 8. 计算总退化面积占比
        total_degraded_ratio = self._calc_total_degraded_ratio(regions, orig_h, orig_w)

        return RegionalDiagnosisResult(
            grid_rows=self.config.grid_rows,
            grid_cols=self.config.grid_cols,
            regions=regions,
            degradation_mask=None,
            total_degraded_ratio=total_degraded_ratio,
        )

    # ============================================================
    # 单区域诊断
    # ============================================================

    def _diagnose_single_region(self,
                                cell_gray: np.ndarray,
                                cell_color: np.ndarray,
                                row: int,
                                col: int,
                                bbox: Tuple[int, int, int, int],
                                mud_bboxes: List[Tuple[int, int, int, int]] = None
                                ) -> RegionDegradationInfo:
        """
        诊断单个网格区域的退化情况
        计算多种退化类型的评分，取置信度最高的作为结果
        """
        # 计算该区域的所有统计特征
        stats = self._calc_region_stats(
            cell_gray, cell_color,
            row, col,
            self.config.grid_rows, self.config.grid_cols,
            mud_bboxes, bbox
        )

        # 计算各退化类型的置信度（0~1）
        scores = {}

        # 过曝
        overexposure_score, overexposure_ratio = self._score_overexposure(stats)
        scores[DegradationType.OVEREXPOSURE] = overexposure_score

        # 低光
        lowlight_score, lowlight_ratio = self._score_low_light(stats)
        scores[DegradationType.LOW_LIGHT] = lowlight_score

        # 雨滴（新版：基于圆形度+局部对比度）
        rain_score, rain_density = self._score_rain(stats)
        scores[DegradationType.RAIN_DROP] = rain_score

        # 泥污（新版：基于颜色+位置先验+纹理粗糙度）
        mud_score = self._score_mud(stats)
        scores[DegradationType.MUD] = mud_score

        # 运动模糊
        blur_score = self._score_blur(stats)
        scores[DegradationType.MOTION_BLUR] = blur_score

        # 取置信度最高的退化类型
        best_type = max(scores, key=scores.get)
        best_confidence = scores[best_type]

        # === 正常区域过滤 ===
        # 如果是正常场景（天空/植被/道路/建筑），大幅降低置信度
        if stats.get('is_normal_scene', False):
            normal_type = stats.get('normal_scene_type', 'unknown')
            # 不同正常类型的抑制强度不同
            suppression = {
                'sky': 0.3,        # 天空：抑制强（天空容易被误判为过曝/低光）
                'vegetation': 0.4, # 植被：抑制中等（树叶容易被误判）
                'road': 0.5,       # 道路：抑制较弱
                'building': 0.6,   # 建筑：抑制较弱
                'unknown': 0.8,
            }
            suppress_factor = suppression.get(normal_type, 0.7)
            best_confidence *= suppress_factor

        # 如果最高置信度低于阈值，视为正常
        if best_confidence < self.config.min_confidence:
            return RegionDegradationInfo(
                grid_row=row,
                grid_col=col,
                bbox=bbox,
                degradation_type=DegradationType.NONE,
                severity=DegradationSeverity.NONE,
                confidence=0.0,
                pixel_ratio=0.0,
                quality_score=1.0,
                stats=stats,
            )

        # 确定严重程度
        severity = self._determine_severity(best_type, best_confidence, stats)

        # 确定像素占比
        pixel_ratio = self._estimate_pixel_ratio(best_type, stats)

        # 质量评分
        quality_score = max(0.0, 1.0 - best_confidence)

        return RegionDegradationInfo(
            grid_row=row,
            grid_col=col,
            bbox=bbox,
            degradation_type=best_type,
            severity=severity,
            confidence=best_confidence,
            pixel_ratio=pixel_ratio,
            quality_score=quality_score,
            stats=stats,
        )

    # ============================================================
    # 区域统计特征计算
    # ============================================================

    def _calc_region_stats(self,
                           cell_gray: np.ndarray,
                           cell_color: np.ndarray,
                           row: int = 0,
                           col: int = 0,
                           grid_rows: int = 6,
                           grid_cols: int = 8,
                           mud_bboxes: List[Tuple[int, int, int, int]] = None,
                           grid_bbox: Tuple[int, int, int, int] = None) -> dict:
        """计算单个区域的所有统计特征"""
        brightness_mean, brightness_std, overexposed_ratio, dark_ratio = \
            utils.calc_brightness_stats(cell_gray)

        laplacian_var = utils.calc_laplacian_variance(cell_gray)
        edge_density = utils.calc_edge_density(cell_gray)
        image_entropy = utils.calc_image_entropy(cell_gray)
        spot_density, spot_count = utils.calc_bright_spot_density(cell_gray)
        color_stats = utils.calc_color_stats(cell_color)

        # === 雨滴核心特征（v4：基于边缘圆形检测） ===
        # 1. 边缘圆形密度（主要特征）- 雨滴边缘是闭合的圆形
        edge_circle_density, edge_circle_count, edge_avg_circularity = \
            utils.detect_rain_circles_edge(
                cell_gray,
                min_radius=2,
                max_radius=12,
                circularity_threshold=self.config.rain_circularity_min,
            )

        # 2. 雨滴边缘密度比（辅助特征）- 小尺度边缘多，大尺度边缘少
        rain_edge_density = utils.calc_rain_edge_density(cell_gray)

        # 3. 局部纹理异常度（辅助特征）
        texture_anomaly = utils.calc_local_texture_anomaly(
            cell_gray, window_size=15
        )

        # 4. 半透明模糊度（辅助特征）
        transparency_blur = utils.calc_transparency_blur_score(
            cell_gray, window_size=11
        )

        # 5. 饱和度降低（关键特征！雨滴折射会让饱和度降低）
        # 树叶、建筑等自然物体饱和度高，雨滴区域饱和度低
        cell_hsv = cv2.cvtColor(cell_color, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(cell_hsv[:, :, 1]) / 255.0  # 归一化到0~1

        # === 泥污核心特征（新版） ===
        # 1. 连通域面积占比（主要特征）- 从整图预计算的连通域中计算
        if mud_bboxes is not None and grid_bbox is not None:
            mud_area_ratio = utils.calc_mud_in_grid(mud_bboxes, grid_bbox)
        else:
            mud_area_ratio = 0.0

        # 2. 颜色匹配度（辅助特征）
        mud_color_ratio = utils.calc_mud_color_score(cell_color)

        # 3. 纹理粗糙度（辅助特征）
        texture_roughness = utils.calc_texture_roughness(cell_gray)

        # === 位置先验 ===
        row_ratio = row / max(grid_rows - 1, 1)  # 0=顶部, 1=底部
        col_ratio = col / max(grid_cols - 1, 1)  # 0=左, 1=右

        # 泥污位置得分：底部 + 边缘 → 得分高
        bottom_score = row_ratio
        edge_score = 1.0 - abs(col_ratio - 0.5) * 2
        edge_score = max(0, edge_score)
        position_score = bottom_score * 0.6 + edge_score * 0.4
        if row_ratio < 0.5:
            position_score *= 0.2  # 上半部分位置权重很低

        # === 正常场景判断 ===
        is_normal, normal_type = utils.is_normal_scene_region(cell_gray, cell_color)

        return {
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'overexposed_ratio': overexposed_ratio,
            'dark_ratio': dark_ratio,
            'laplacian_var': laplacian_var,
            'edge_density': edge_density,
            'image_entropy': image_entropy,
            'spot_density': spot_density,
            'spot_count': spot_count,
            'saturation_mean': color_stats['s_mean'],
            'value_mean': color_stats['v_mean'],
            # 雨滴新特征（v4：边缘圆形检测）
            'edge_circle_density': edge_circle_density,
            'edge_circle_count': edge_circle_count,
            'edge_avg_circularity': edge_avg_circularity,
            'rain_edge_density': rain_edge_density,
            'texture_anomaly': texture_anomaly,
            'transparency_blur': transparency_blur,
            'avg_saturation': avg_saturation,  # 平均饱和度（关键！雨滴饱和度低）
            # 雨滴旧特征（保留）
            'spot_density': spot_density,
            'spot_count': spot_count,
            # 泥污新特征
            'mud_area_ratio': mud_area_ratio,
            'mud_color_ratio': mud_color_ratio,
            'texture_roughness': texture_roughness,
            # 位置信息
            'row_ratio': row_ratio,
            'col_ratio': col_ratio,
            'mud_position_score': position_score,
            # 正常场景
            'is_normal_scene': is_normal,
            'normal_scene_type': normal_type,
        }

    # ============================================================
    # 各退化类型评分（返回0~1置信度，越高越可能是该类型）
    # ============================================================

    def _score_overexposure(self, stats: dict) -> Tuple[float, float]:
        """
        过曝评分
        高过曝像素占比 + 低边缘密度 → 高置信度
        """
        cfg = self.config
        ratio = stats['overexposed_ratio']

        # 过曝像素占比评分
        if ratio >= cfg.overexposure_severe_ratio:
            ratio_score = 1.0
        elif ratio <= 0.05:
            ratio_score = 0.0
        else:
            ratio_score = (ratio - 0.05) / (cfg.overexposure_severe_ratio - 0.05)

        # 边缘密度验证（过曝区域边缘少）
        edge = stats['edge_density']
        if edge <= cfg.overexposure_edge_density_max:
            edge_score = 1.0
        elif edge >= 0.05:
            edge_score = 0.0
        else:
            edge_score = 1.0 - (edge - cfg.overexposure_edge_density_max) / (0.05 - cfg.overexposure_edge_density_max)

        # 综合
        confidence = ratio_score * 0.6 + edge_score * 0.4
        return max(0.0, min(1.0, confidence)), ratio

    def _score_low_light(self, stats: dict) -> Tuple[float, float]:
        """
        低光评分
        高暗像素占比 + 低亮度 → 高置信度
        """
        cfg = self.config
        ratio = stats['dark_ratio']

        # 暗像素占比评分
        if ratio >= cfg.lowlight_severe_ratio:
            ratio_score = 1.0
        elif ratio <= 0.1:
            ratio_score = 0.0
        else:
            ratio_score = (ratio - 0.1) / (cfg.lowlight_severe_ratio - 0.1)

        # 亮度评分
        brightness = stats['brightness_mean']
        if brightness <= cfg.lowlight_brightness_max:
            brightness_score = 1.0
        elif brightness >= 100:
            brightness_score = 0.0
        else:
            brightness_score = 1.0 - (brightness - cfg.lowlight_brightness_max) / (100 - cfg.lowlight_brightness_max)

        confidence = ratio_score * 0.5 + brightness_score * 0.5
        return max(0.0, min(1.0, confidence)), ratio

    def _score_rain(self, stats: dict) -> Tuple[float, float]:
        """
        雨滴评分（v5版：基于边缘圆形检测）
        """
        cfg = self.config

        # 1. 边缘圆形密度评分（主要特征，权重0.6）
        edge_circle_density = stats.get('edge_circle_density', 0)
        if edge_circle_density >= cfg.rain_severe_edge_circle_density:
            circle_score = 1.0
        elif edge_circle_density <= cfg.rain_mild_edge_circle_density:
            circle_score = 0.0
        else:
            circle_score = (edge_circle_density - cfg.rain_mild_edge_circle_density) / \
                           (cfg.rain_severe_edge_circle_density - cfg.rain_mild_edge_circle_density)

        # 2. 雨滴边缘密度比评分（权重0.2）
        rain_edge_density = stats.get('rain_edge_density', 0)
        edge_score = min(1.0, rain_edge_density)

        # 3. 局部纹理异常度评分（权重0.1）
        texture_anomaly = stats.get('texture_anomaly', 0)
        anomaly_score = min(1.0, texture_anomaly * 2.0)

        # 4. 半透明模糊度评分（权重0.1）
        transparency_blur = stats.get('transparency_blur', 0)
        blur_score = min(1.0, transparency_blur * 1.5)

        # 综合加权
        confidence = (
            circle_score * 0.6 +
            edge_score * 0.2 +
            anomaly_score * 0.1 +
            blur_score * 0.1
        )

        # 天空区域加成
        if stats.get('normal_scene_type') == 'sky':
            confidence = min(1.0, confidence * 1.2)

        # 植被区域抑制
        if stats.get('normal_scene_type') == 'vegetation':
            confidence *= 0.3

        return max(0.0, min(1.0, confidence)), edge_circle_density

    def _score_mud(self, stats: dict) -> float:
        """
        泥污评分（新版 v3）
        核心特征：连通域面积占比 + 颜色匹配度 + 位置先验

        泥污的真正特征：
        1. 大面积连续的污渍区域（不是分散的像素）
        2. 位置：几乎只在画面底部（飞溅的泥水）
        3. 颜色：深褐、黑褐、黄褐色
        """
        cfg = self.config

        # 1. 连通域面积占比评分（主要特征，权重0.5）
        # 泥污是大面积连续区域
        mud_area_ratio = stats.get('mud_area_ratio', 0)
        if mud_area_ratio >= cfg.mud_severe_area_ratio:
            area_score = 1.0
        elif mud_area_ratio <= cfg.mud_mild_area_ratio:
            area_score = 0.0
        else:
            area_score = (mud_area_ratio - cfg.mud_mild_area_ratio) / \
                         (cfg.mud_severe_area_ratio - cfg.mud_mild_area_ratio)

        # 2. 颜色匹配度评分（辅助特征，权重0.3）
        mud_color_ratio = stats.get('mud_color_ratio', 0)
        if mud_color_ratio >= cfg.mud_color_severe_ratio:
            color_score = 1.0
        elif mud_color_ratio <= 0.05:
            color_score = 0.0
        else:
            color_score = (mud_color_ratio - 0.05) / (cfg.mud_color_severe_ratio - 0.05)

        # 3. 位置先验评分（强先验，权重0.2）
        # 泥污几乎都在底部
        position_score = stats.get('mud_position_score', 0)

        # 综合加权
        confidence = (
            area_score * 0.5 +
            color_score * 0.3 +
            position_score * 0.2
        )

        # 额外：如果是道路区域，泥污置信度加成
        if stats.get('normal_scene_type') == 'road':
            confidence = min(1.0, confidence * 1.1)

        # 额外：植被区域泥污置信度大幅降低（树叶颜色容易被误判为泥污）
        if stats.get('normal_scene_type') == 'vegetation':
            confidence *= 0.2

        # 额外：上半区域泥污置信度大幅降低（不太可能）
        if stats.get('row_ratio', 0) < 0.5:
            confidence *= 0.3

        return max(0.0, min(1.0, confidence))

    def _score_blur(self, stats: dict) -> float:
        """
        运动模糊评分（已禁用）
        当前版本只识别：雨滴、泥污、过曝、低光
        """
        return 0.0

    # ============================================================
    # 严重程度判定
    # ============================================================

    def _determine_severity(self,
                            degradation_type: DegradationType,
                            confidence: float,
                            stats: dict) -> DegradationSeverity:
        """根据退化类型和置信度/特征值确定严重程度"""
        cfg = self.config

        if degradation_type == DegradationType.OVEREXPOSURE:
            ratio = stats['overexposed_ratio']
            if ratio >= cfg.overexposure_severe_ratio:
                return DegradationSeverity.SEVERE
            elif ratio >= cfg.overexposure_moderate_ratio:
                return DegradationSeverity.MODERATE
            else:
                return DegradationSeverity.MILD

        elif degradation_type == DegradationType.LOW_LIGHT:
            ratio = stats['dark_ratio']
            if ratio >= cfg.lowlight_severe_ratio:
                return DegradationSeverity.SEVERE
            elif ratio >= cfg.lowlight_moderate_ratio:
                return DegradationSeverity.MODERATE
            else:
                return DegradationSeverity.MILD

        elif degradation_type == DegradationType.RAIN_DROP:
            # 雨滴严重程度基于边缘圆形密度
            density = stats.get('edge_circle_density', 0)
            if density >= cfg.rain_severe_edge_circle_density:
                return DegradationSeverity.SEVERE
            elif density >= (cfg.rain_mild_edge_circle_density + cfg.rain_severe_edge_circle_density) / 2:
                return DegradationSeverity.MODERATE
            else:
                return DegradationSeverity.MILD

        elif degradation_type == DegradationType.MUD:
            # 泥污严重程度基于连通域面积占比
            area_ratio = stats.get('mud_area_ratio', 0)
            if area_ratio >= cfg.mud_severe_area_ratio:
                return DegradationSeverity.SEVERE
            elif area_ratio >= (cfg.mud_mild_area_ratio + cfg.mud_severe_area_ratio) / 2:
                return DegradationSeverity.MODERATE
            else:
                return DegradationSeverity.MILD

        elif degradation_type == DegradationType.MOTION_BLUR:
            laplacian = stats['laplacian_var']
            if laplacian <= cfg.blur_severe_laplacian:
                return DegradationSeverity.SEVERE
            elif laplacian <= cfg.blur_moderate_laplacian:
                return DegradationSeverity.MODERATE
            else:
                return DegradationSeverity.MILD

        return DegradationSeverity.MILD

    def _estimate_pixel_ratio(self,
                              degradation_type: DegradationType,
                              stats: dict) -> float:
        """估算退化像素在该区域中的占比"""
        if degradation_type == DegradationType.OVEREXPOSURE:
            return stats['overexposed_ratio']
        elif degradation_type == DegradationType.LOW_LIGHT:
            return stats['dark_ratio']
        elif degradation_type == DegradationType.RAIN_DROP:
            # 雨滴覆盖比例：用边缘圆形密度估算
            density = stats.get('edge_circle_density', 0)
            # 假设密度1.5对应40%覆盖，密度0.3对应10%覆盖
            coverage = min(0.6, density * 0.3)
            return max(0.0, coverage)
        elif degradation_type == DegradationType.MUD:
            # 泥污覆盖比例 = 连通域面积占比
            return stats.get('mud_area_ratio', 0.3)
        elif degradation_type == DegradationType.MOTION_BLUR:
            # 模糊是全局性的，假设覆盖整个区域
            return 1.0
        return 0.5

    # ============================================================
    # 泥污时序稳定性验证
    # ============================================================

    def _verify_mud_temporal_stability(self,
                                       current_regions: List[RegionDegradationInfo]
                                       ) -> List[RegionDegradationInfo]:
        """
        验证泥污的时序稳定性
        泥污应该是静态的，如果连续N帧都检测到泥污才确认，
        否则可能是雨滴或其他动态退化。
        """
        cfg = self.config
        stable_frames = cfg.mud_temporal_stable_frames

        # 历史帧数不足，降低泥污置信度
        if len(self._history_buffer) < stable_frames - 1:
            result = []
            for region in current_regions:
                if region.degradation_type == DegradationType.MUD:
                    # 帧数不足，泥污置信度打折
                    new_region = RegionDegradationInfo(
                        grid_row=region.grid_row,
                        grid_col=region.grid_col,
                        bbox=region.bbox,
                        degradation_type=DegradationType.MUD,
                        severity=region.severity,
                        confidence=region.confidence * 0.6,  # 暂时降低置信度
                        pixel_ratio=region.pixel_ratio,
                        quality_score=region.quality_score,
                        stats=region.stats,
                    )
                    result.append(new_region)
                else:
                    result.append(region)
            return result

        # 检查最近 N 帧中该网格是否持续为泥污
        history = list(self._history_buffer)[-(stable_frames - 1):]

        result = []
        for i, region in enumerate(current_regions):
            if region.degradation_type != DegradationType.MUD:
                result.append(region)
                continue

            # 检查历史帧中该网格是否也是泥污
            mud_count = 1  # 当前帧
            for hist_frame in history:
                if i < len(hist_frame):
                    if hist_frame[i].degradation_type == DegradationType.MUD:
                        mud_count += 1

            if mud_count >= stable_frames:
                # 确认是泥污，保持高置信度
                result.append(region)
            else:
                # 不稳定，降级为雨滴或降低置信度
                # 如果亮斑密度也高，可能是雨滴；否则降低泥污置信度
                spot_density = region.stats.get('spot_density', 0)
                if spot_density > 2.0:
                    # 更可能是雨滴
                    new_region = RegionDegradationInfo(
                        grid_row=region.grid_row,
                        grid_col=region.grid_col,
                        bbox=region.bbox,
                        degradation_type=DegradationType.RAIN_DROP,
                        severity=DegradationSeverity.MILD,
                        confidence=min(region.confidence, 0.6),
                        pixel_ratio=region.pixel_ratio * 0.5,
                        quality_score=max(0.0, 1.0 - min(region.confidence, 0.6)),
                        stats=region.stats,
                    )
                    result.append(new_region)
                else:
                    # 不确定，降低置信度
                    new_region = RegionDegradationInfo(
                        grid_row=region.grid_row,
                        grid_col=region.grid_col,
                        bbox=region.bbox,
                        degradation_type=DegradationType.MUD,
                        severity=region.severity,
                        confidence=region.confidence * 0.5,
                        pixel_ratio=region.pixel_ratio,
                        quality_score=region.quality_score,
                        stats=region.stats,
                    )
                    result.append(new_region)

        return result

    # ============================================================
    # 辅助方法
    # ============================================================

    def _calc_total_degraded_ratio(self,
                                   regions: List[RegionDegradationInfo],
                                   img_h: int,
                                   img_w: int) -> float:
        """计算总退化面积占比"""
        total_area = img_h * img_w
        degraded_area = 0

        for region in regions:
            if region.degradation_type != DegradationType.NONE:
                cell_area = (region.bbox[2] - region.bbox[0]) * \
                            (region.bbox[3] - region.bbox[1])
                degraded_area += cell_area * region.pixel_ratio

        return degraded_area / total_area if total_area > 0 else 0.0

    def generate_degradation_mask(self,
                                   image: np.ndarray,
                                   regional_result: RegionalDiagnosisResult
                                   ) -> np.ndarray:
        """
        生成像素级退化掩码（可视化用）
        将网格级别的结果上采样到像素级别
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        for region in regional_result.regions:
            if region.degradation_type == DegradationType.NONE:
                continue
            x1, y1, x2, y2 = region.bbox
            mask[y1:y2, x1:x2] = region.confidence

        # 高斯平滑，让边界更自然
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask
