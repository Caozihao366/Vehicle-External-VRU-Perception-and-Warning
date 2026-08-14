"""
传感器质量检测系统 - 第一级：全局快速体检
对整图进行快速统计特征计算，判断图像是否存在退化。
正常帧直接放行，异常帧触发第二级精细诊断。

核心设计：
- 纯统计方法，无学习，极快（目标<2ms）
- 多维度并行计算：过曝、低光、模糊、雨滴初筛
- 输出全局质量评分 + 疑似退化类型
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

from .data_types import (
    DegradationType,
    GlobalQualityStatus,
    GlobalQualityResult,
)
from . import utils


# ============================================================
# 配置参数
# ============================================================

@dataclass
class GlobalCheckerConfig:
    """全局质量检测器配置"""
    # 降采样尺寸（越小越快，但精度降低）
    downsample_max_size: int = 480

    # 过曝检测阈值
    overexposed_ratio_threshold: float = 0.15  # 过曝像素占比超过此值视为过曝
    overexposure_edge_density_max: float = 0.01  # 过曝区域边缘密度上限

    # 低光检测阈值
    dark_ratio_threshold: float = 0.40         # 暗像素占比超过此值视为低光
    low_light_brightness_threshold: float = 60.0  # 平均亮度低于此值视为低光

    # 模糊检测阈值
    blur_laplacian_var_threshold: float = 100.0  # 拉普拉斯方差低于此值视为模糊
    blur_gradient_median_threshold: float = 15.0  # 梯度中位数低于此值视为模糊

    # 雨滴初筛阈值
    rain_spot_density_threshold: float = 2.0   # 亮斑密度（个/千像素）
    rain_entropy_drop_threshold: float = 1.0   # 熵下降幅度

    # 质量评分阈值
    quality_score_degraded: float = 0.6        # 低于此值视为整体退化
    quality_score_partial: float = 0.8         # 低于此值视为局部退化

    # 正常场景基线（用于自适应阈值，可在线更新）
    normal_entropy_baseline: float = 6.5


# ============================================================
# 全局质量检测器
# ============================================================

class GlobalQualityChecker:
    """
    第一级：全局快速体检
    输入：原始图像
    输出：GlobalQualityResult（全局质量评分 + 疑似退化类型）
    """

    def __init__(self, config: GlobalCheckerConfig = None):
        self.config = config or GlobalCheckerConfig()
        self._baseline_entropy = self.config.normal_entropy_baseline
        self._baseline_update_count = 0

    def check(self, image: np.ndarray) -> GlobalQualityResult:
        """
        执行全局质量检测
        Args:
            image: 输入图像 (BGR格式)
        Returns:
            GlobalQualityResult
        """
        # 1. 降采样加速
        img_small = utils.downsample(image, self.config.downsample_max_size)
        gray = utils.to_grayscale(img_small)

        # 2. 并行计算各维度特征
        brightness_mean, brightness_std, overexposed_ratio, dark_ratio = \
            utils.calc_brightness_stats(gray)

        laplacian_var = utils.calc_laplacian_variance(gray)
        gradient_median = utils.calc_gradient_median(gray)
        edge_density = utils.calc_edge_density(gray)
        image_entropy = utils.calc_image_entropy(gray)
        spot_density, spot_count = utils.calc_bright_spot_density(gray)

        # 3. 各维度退化判定
        suspected_types = []
        dimension_scores = {}

        # 过曝检测
        overexposure_score = self._check_overexposure(overexposed_ratio, gray)
        dimension_scores['overexposure'] = overexposure_score
        if overexposure_score < 0.7:
            suspected_types.append(DegradationType.OVEREXPOSURE)

        # 低光检测
        lowlight_score = self._check_low_light(dark_ratio, brightness_mean)
        dimension_scores['lowlight'] = lowlight_score
        if lowlight_score < 0.7:
            suspected_types.append(DegradationType.LOW_LIGHT)

        # 模糊检测
        blur_score = self._check_blur(laplacian_var, gradient_median)
        dimension_scores['blur'] = blur_score
        if blur_score < 0.7:
            suspected_types.append(DegradationType.MOTION_BLUR)

        # 雨滴初筛
        rain_score = self._check_rain_initial(spot_density, image_entropy)
        dimension_scores['rain'] = rain_score
        if rain_score < 0.7:
            suspected_types.append(DegradationType.RAIN_DROP)

        # 4. 综合质量评分（取各维度最差的）
        quality_score = min(dimension_scores.values()) if dimension_scores else 1.0

        # 5. 判定全局状态
        if quality_score >= self.config.quality_score_partial:
            status = GlobalQualityStatus.NORMAL
        elif quality_score >= self.config.quality_score_degraded:
            status = GlobalQualityStatus.PARTIALLY_DEGRADED
        else:
            status = GlobalQualityStatus.DEGRADED

        is_degraded = len(suspected_types) > 0 and quality_score < 0.9

        # 6. 组装结果
        result = GlobalQualityResult(
            quality_score=quality_score,
            status=status,
            suspected_types=suspected_types,
            is_degraded=is_degraded,
            brightness_mean=brightness_mean,
            brightness_std=brightness_std,
            overexposed_ratio=overexposed_ratio,
            dark_ratio=dark_ratio,
            laplacian_variance=laplacian_var,
            gradient_median=gradient_median,
            edge_density=edge_density,
            image_entropy=image_entropy,
            bright_spot_density=spot_density,
            bright_spot_count=spot_count,
        )

        return result

    # ============================================================
    # 各维度检测方法（返回0~1评分，1=正常，0=完全退化）
    # ============================================================

    def _check_overexposure(self,
                            overexposed_ratio: float,
                            gray: np.ndarray) -> float:
        """
        过曝检测
        逻辑：过曝像素占比高 + 边缘密度低 → 确认过曝
        """
        cfg = self.config

        # 过曝像素占比评分
        if overexposed_ratio <= 0.05:
            ratio_score = 1.0
        elif overexposed_ratio >= cfg.overexposed_ratio_threshold:
            ratio_score = 0.0
        else:
            ratio_score = 1.0 - (overexposed_ratio - 0.05) / (cfg.overexposed_ratio_threshold - 0.05)

        # 边缘密度验证（真正的过曝区域边缘很少）
        edge_density = utils.calc_edge_density(gray)
        if edge_density <= cfg.overexposure_edge_density_max:
            edge_score = 0.0  # 边缘极少，确认过曝
        elif edge_density >= 0.05:
            edge_score = 1.0
        else:
            edge_score = (edge_density - cfg.overexposure_edge_density_max) / (0.05 - cfg.overexposure_edge_density_max)

        # 综合：两个条件都指向过曝才判定为过曝
        # 如果过曝像素多但边缘也多，可能是白色物体而非过曝
        combined = ratio_score * 0.6 + edge_score * 0.4
        return max(0.0, min(1.0, combined))

    def _check_low_light(self,
                         dark_ratio: float,
                         brightness_mean: float) -> float:
        """
        低光检测
        逻辑：暗像素占比高 + 平均亮度低 → 确认低光
        """
        cfg = self.config

        # 暗像素占比评分
        if dark_ratio <= 0.1:
            ratio_score = 1.0
        elif dark_ratio >= cfg.dark_ratio_threshold:
            ratio_score = 0.0
        else:
            ratio_score = 1.0 - (dark_ratio - 0.1) / (cfg.dark_ratio_threshold - 0.1)

        # 平均亮度评分
        if brightness_mean >= 100:
            brightness_score = 1.0
        elif brightness_mean <= cfg.low_light_brightness_threshold:
            brightness_score = 0.0
        else:
            brightness_score = (brightness_mean - cfg.low_light_brightness_threshold) / (100 - cfg.low_light_brightness_threshold)

        # 综合
        combined = ratio_score * 0.5 + brightness_score * 0.5
        return max(0.0, min(1.0, combined))

    def _check_blur(self,
                    laplacian_var: float,
                    gradient_median: float) -> float:
        """
        模糊检测
        逻辑：拉普拉斯方差低 + 梯度中位数低 → 确认模糊
        """
        cfg = self.config

        # 拉普拉斯方差评分
        if laplacian_var >= cfg.blur_laplacian_var_threshold * 2:
            laplacian_score = 1.0
        elif laplacian_var <= cfg.blur_laplacian_var_threshold:
            laplacian_score = 0.0
        else:
            laplacian_score = (laplacian_var - cfg.blur_laplacian_var_threshold) / \
                              (cfg.blur_laplacian_var_threshold * 2 - cfg.blur_laplacian_var_threshold)

        # 梯度中位数评分
        if gradient_median >= cfg.blur_gradient_median_threshold * 2:
            gradient_score = 1.0
        elif gradient_median <= cfg.blur_gradient_median_threshold:
            gradient_score = 0.0
        else:
            gradient_score = (gradient_median - cfg.blur_gradient_median_threshold) / \
                             (cfg.blur_gradient_median_threshold * 2 - cfg.blur_gradient_median_threshold)

        # 综合
        combined = laplacian_score * 0.6 + gradient_score * 0.4
        return max(0.0, min(1.0, combined))

    def _check_rain_initial(self,
                            spot_density: float,
                            image_entropy: float) -> float:
        """
        雨滴初筛
        逻辑：亮斑密度高 + 图像熵下降 → 疑似雨滴
        """
        cfg = self.config

        # 亮斑密度评分
        if spot_density <= 0.5:
            spot_score = 1.0
        elif spot_density >= cfg.rain_spot_density_threshold:
            spot_score = 0.0
        else:
            spot_score = 1.0 - (spot_density - 0.5) / (cfg.rain_spot_density_threshold - 0.5)

        # 熵下降评分
        entropy_drop = self._baseline_entropy - image_entropy
        if entropy_drop <= 0:
            entropy_score = 1.0
        elif entropy_drop >= cfg.rain_entropy_drop_threshold:
            entropy_score = 0.0
        else:
            entropy_score = 1.0 - entropy_drop / cfg.rain_entropy_drop_threshold

        # 综合
        combined = spot_score * 0.6 + entropy_score * 0.4
        return max(0.0, min(1.0, combined))

    # ============================================================
    # 基线更新（自适应当前场景）
    # ============================================================

    def update_baseline(self, result: GlobalQualityResult):
        """
        在线更新正常场景基线
        仅在判定为正常时更新，用滑动平均
        """
        if result.status == GlobalQualityStatus.NORMAL:
            alpha = 0.01  # 滑动平均系数
            self._baseline_entropy = (1 - alpha) * self._baseline_entropy + alpha * result.image_entropy
            self._baseline_update_count += 1
