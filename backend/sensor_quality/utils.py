"""
传感器质量检测系统 - 工具函数模块
包含图像统计特征计算、网格划分、可视化等通用工具
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

from .data_types import DegradationType, DegradationSeverity, BlindZoneInfo, RegionDegradationInfo


# ============================================================
# 基础图像处理
# ============================================================

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """转换为灰度图"""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def downsample(image: np.ndarray, max_size: int = 640) -> np.ndarray:
    """等比例降采样到 max_size（保持宽高比）"""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ============================================================
# 亮度与对比度统计
# ============================================================

def calc_brightness_stats(gray: np.ndarray) -> Tuple[float, float, float, float]:
    """
    计算亮度统计特征
    Returns:
        (均值, 标准差, 过曝像素占比(>240), 暗像素占比(<30))
    """
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    overexposed_ratio = float(np.sum(gray > 240) / gray.size)
    dark_ratio = float(np.sum(gray < 30) / gray.size)
    return mean_val, std_val, overexposed_ratio, dark_ratio


# ============================================================
# 清晰度 / 模糊检测
# ============================================================

def calc_laplacian_variance(gray: np.ndarray) -> float:
    """
    计算拉普拉斯方差（清晰度指标）
    值越小越模糊
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def calc_gradient_median(gray: np.ndarray) -> float:
    """
    计算梯度幅值中位数（清晰度指标）
    使用 Sobel 算子
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return float(np.median(grad_mag))


def calc_edge_density(gray: np.ndarray, threshold: int = 50) -> float:
    """
    计算边缘密度（Canny边缘像素占比）
    过曝/低光区域边缘密度极低
    """
    edges = cv2.Canny(gray, threshold, threshold * 2)
    return float(np.sum(edges > 0) / gray.size)


# ============================================================
# 信息量 / 纹理统计
# ============================================================

def calc_image_entropy(gray: np.ndarray) -> float:
    """
    计算图像熵（信息量指标）
    雨滴、模糊等会导致熵下降
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # 归一化
    hist = hist[hist > 0]     # 去掉0值避免log(0)
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


# ============================================================
# 亮斑检测（雨滴初筛）
# ============================================================

def calc_bright_spot_density(gray: np.ndarray,
                              threshold: int = 220,
                              min_area: int = 3,
                              max_area: int = 100) -> Tuple[float, int]:
    """
    计算亮斑密度（雨滴/水珠初筛）
    Args:
        gray: 灰度图
        threshold: 亮度阈值
        min_area: 最小亮斑面积（像素）
        max_area: 最大亮斑面积（像素）
    Returns:
        (亮斑密度（个/千像素）, 亮斑数量)
    """
    # 二值化
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 形态学开运算去噪
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 筛选符合面积范围的亮斑
    spot_count = 0
    for i in range(1, num_labels):  # 跳过背景(0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            spot_count += 1

    density = spot_count / (gray.size / 1000.0)
    return density, spot_count


# ============================================================
# 圆形度检测（雨滴核心特征）
# ============================================================

def calc_circular_spot_density(gray: np.ndarray,
                                min_radius: int = 2,
                                max_radius: int = 15,
                                circularity_threshold: float = 0.6) -> Tuple[float, int, float]:
    """
    计算圆形亮斑密度（雨滴检测的核心特征）
    通过连通域形状分析，找出接近圆形的亮斑

    雨滴的关键特征：
    1. 形状接近圆形（圆形度高）
    2. 大小在一定范围内
    3. 边缘有明显梯度（折射效果）

    Args:
        gray: 灰度图
        min_radius: 最小半径（像素）
        max_radius: 最大半径（像素）
        circularity_threshold: 圆形度阈值（0~1，越接近1越圆）
    Returns:
        (圆形亮斑密度（个/千像素）, 圆形亮斑数量, 平均圆形度)
    """
    h, w = gray.shape

    # 自适应阈值找局部亮斑（比固定阈值更鲁棒）
    # 用局部均值 + 偏移作为阈值
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=5
    )

    # 形态学操作去噪
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    circular_count = 0
    circularity_sum = 0.0

    min_area = np.pi * min_radius * min_radius
    max_area = np.pi * max_radius * max_radius

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # 面积筛选
        if area < min_area or area > max_area:
            continue

        # 计算圆形度：4πA / P²
        # 完美圆形的圆形度=1，越不规则越小
        # 找轮廓
        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity >= circularity_threshold:
            circular_count += 1
            circularity_sum += circularity

    avg_circularity = circularity_sum / circular_count if circular_count > 0 else 0.0
    density = circular_count / (gray.size / 1000.0)

    return density, circular_count, avg_circularity


def calc_local_contrast_variation(gray: np.ndarray,
                                   kernel_size: int = 7) -> float:
    """
    计算局部对比度变化（雨滴折射导致的纹理变化）
    雨滴区域的局部对比度会异常变化（折射变形）

    方法：计算局部标准差图，再计算其方差
    正常区域：局部标准差相对均匀
    雨滴区域：局部标准差变化剧烈

    Args:
        gray: 灰度图
        kernel_size: 局部窗口大小
    Returns:
        局部对比度变化程度（0~1归一化）
    """
    # 计算局部均值
    local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))

    # 计算局部方差
    local_sq_mean = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)

    # 计算局部标准差的方差（衡量对比度变化的剧烈程度）
    std_variation = float(np.var(local_std))

    # 归一化（经验值）
    normalized = min(1.0, std_variation / 500.0)
    return normalized


# ============================================================
# 泥污颜色检测
# ============================================================

def calc_mud_color_score(image: np.ndarray) -> float:
    """
    计算泥污颜色匹配度
    泥污的典型颜色：黄褐色、暗棕色、灰褐色

    HSV颜色范围：
    - 色相(H): 10~40度（黄到橙红）
    - 饱和度(S): 20~120（中等偏低，偏灰）
    - 明度(V): 30~150（偏暗）

    Args:
        image: BGR图像
    Returns:
        泥污颜色匹配度（0~1，越高越像泥污）
    """
    if len(image.shape) == 2:
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 泥污颜色范围（OpenCV中H是0~179）
    # 黄褐色：H 10~30, S 20~100, V 40~150
    lower_mud1 = np.array([10, 20, 40])
    upper_mud1 = np.array([30, 120, 160])

    # 深棕色/灰褐色：H 0~15, S 10~80, V 20~100
    lower_mud2 = np.array([0, 10, 20])
    upper_mud2 = np.array([15, 90, 120])

    mask1 = cv2.inRange(hsv, lower_mud1, upper_mud1)
    mask2 = cv2.inRange(hsv, lower_mud2, upper_mud2)
    mud_mask = cv2.bitwise_or(mask1, mask2)

    # 泥污像素占比
    mud_ratio = float(np.sum(mud_mask > 0) / mud_mask.size)

    return mud_ratio


def calc_texture_roughness(gray: np.ndarray,
                            window_size: int = 5) -> float:
    """
    计算纹理粗糙度（泥污的斑驳纹理）
    泥污不是均匀的，而是斑驳、粗糙的纹理

    方法：计算局部标准差的均值
    - 均匀区域：局部标准差小 → 粗糙度低
    - 斑驳区域：局部标准差大 → 粗糙度高

    Args:
        gray: 灰度图
        window_size: 局部窗口大小
    Returns:
        纹理粗糙度（归一化到0~1）
    """
    # 局部均值
    local_mean = cv2.blur(gray.astype(np.float32), (window_size, window_size))

    # 局部方差
    local_sq_mean = cv2.blur((gray.astype(np.float32) ** 2), (window_size, window_size))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)

    # 平均局部标准差
    avg_local_std = float(np.mean(local_std))

    # 归一化（经验值：正常道路纹理约10~30，泥污斑驳纹理约30~80）
    roughness = min(1.0, avg_local_std / 50.0)
    return roughness


# ============================================================
# 雨滴核心特征：梯度方向熵（折射变形导致方向杂乱）
# ============================================================

def calc_gradient_direction_entropy(gray: np.ndarray,
                                     window_size: int = 15,
                                     num_bins: int = 18) -> float:
    """
    计算局部梯度方向熵（雨滴检测的核心特征）

    雨滴的关键特征：折射变形导致局部梯度方向杂乱
    - 正常区域：梯度方向有规律（边缘、纹理方向一致）→ 熵低
    - 雨滴区域：折射导致梯度方向杂乱无章 → 熵高

    方法：
    1. 计算每个像素的梯度方向（0~180度）
    2. 在局部窗口内统计梯度方向直方图
    3. 计算直方图的熵
    4. 返回平均熵值

    Args:
        gray: 灰度图
        window_size: 局部窗口大小
        num_bins: 方向直方图的bin数
    Returns:
        平均梯度方向熵（归一化到0~1）
    """
    h, w = gray.shape
    gray_float = gray.astype(np.float32)

    # 1. 计算梯度
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)

    # 2. 计算梯度方向（0~180度，因为边缘方向是对称的）
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(np.abs(grad_y), np.abs(grad_x)) * 180 / np.pi  # 0~90度
    # 扩展到0~180度（考虑x和y的符号）
    direction_full = np.arctan2(grad_y, grad_x) * 180 / np.pi  # -180~180
    direction_full = direction_full % 180  # 0~180

    # 只保留有一定梯度幅值的像素（弱梯度噪声忽略）
    mag_threshold = 5.0
    valid_mask = magnitude > mag_threshold

    # 3. 计算全局梯度方向熵（简化版：用整图的方向分布）
    # 为了速度，先计算整图的方向熵
    valid_directions = direction_full[valid_mask]

    if len(valid_directions) < 100:
        return 0.0

    # 统计直方图
    hist, _ = np.histogram(valid_directions, bins=num_bins, range=(0, 180))
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()

    if hist_sum == 0:
        return 0.0

    hist_norm = hist / hist_sum
    hist_norm = hist_norm[hist_norm > 0]  # 去掉0值

    # 计算熵
    entropy = -np.sum(hist_norm * np.log2(hist_norm))

    # 归一化：最大熵 = log2(num_bins)
    max_entropy = np.log2(num_bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return float(normalized_entropy)


def calc_local_gradient_entropy_map(gray: np.ndarray,
                                     window_size: int = 15,
                                     num_bins: int = 12) -> np.ndarray:
    """
    计算局部梯度方向熵图（像素级）
    用于更精确的雨滴区域定位

    注意：计算量较大，建议在降采样后的图像上使用
    """
    h, w = gray.shape
    gray_float = gray.astype(np.float32)

    # 计算梯度方向
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
    direction = direction % 180

    # 量化方向到bin
    direction_bins = np.clip((direction / 180 * num_bins).astype(int), 0, num_bins - 1)
    direction_bins[magnitude < 3] = -1  # 弱梯度标记为无效

    # 用积分图加速计算
    # 为每个bin创建一张积分图
    integral_maps = []
    for b in range(num_bins):
        bin_mask = (direction_bins == b).astype(np.float32)
        integral = cv2.integral(bin_mask)
        integral_maps.append(integral)

    # 计算每个像素的局部熵
    entropy_map = np.zeros((h, w), dtype=np.float32)
    half_w = window_size // 2

    for y in range(half_w, h - half_w, max(1, window_size // 2)):
        for x in range(half_w, w - half_w, max(1, window_size // 2)):
            x1, y1 = x - half_w, y - half_w
            x2, y2 = x + half_w, y + half_w

            # 计算窗口内每个bin的像素数
            counts = []
            total = 0
            for b in range(num_bins):
                integral = integral_maps[b]
                count = (integral[y2+1, x2+1] - integral[y1, x2+1] -
                         integral[y2+1, x1] + integral[y1, x1])
                counts.append(max(0, count))
                total += count

            if total < 10:
                continue

            # 计算熵
            counts_arr = np.array(counts, dtype=np.float32)
            counts_norm = counts_arr / total
            counts_norm = counts_norm[counts_norm > 0]
            entropy = -np.sum(counts_norm * np.log2(counts_norm))
            max_entropy = np.log2(num_bins)
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

            entropy_map[y-half_w:y+half_w, x-half_w:x+half_w] = norm_entropy

    return entropy_map


# ============================================================
# 雨滴核心特征：局部纹理异常度
# ============================================================

def calc_local_texture_anomaly(gray: np.ndarray,
                                window_size: int = 15) -> float:
    """
    计算局部纹理异常度（雨滴检测的辅助特征）

    雨滴区域的纹理与周围正常区域不同：
    - 雨滴内部：折射导致模糊，纹理变弱
    - 雨滴边缘：折射导致变形，纹理异常
    - 整体：局部方差的分布不均匀

    方法：
    1. 计算局部标准差图
    2. 计算局部标准差的方差（衡量纹理变化的剧烈程度）
    3. 异常度越高，越可能是雨滴

    Args:
        gray: 灰度图
        window_size: 局部窗口大小
    Returns:
        纹理异常度（归一化到0~1）
    """
    gray_float = gray.astype(np.float32)

    # 局部均值
    local_mean = cv2.blur(gray_float, (window_size, window_size))

    # 局部方差
    local_sq_mean = cv2.blur(gray_float ** 2, (window_size, window_size))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)

    # 计算局部标准差的方差（纹理变化的剧烈程度）
    # 雨滴区域：局部标准差变化剧烈（有的地方清晰有的地方模糊）
    std_of_std = float(np.std(local_std))

    # 归一化（经验值）
    anomaly = min(1.0, std_of_std / 30.0)
    return anomaly


# ============================================================
# 雨滴核心特征：半透明模糊度
# ============================================================

def calc_transparency_blur_score(gray: np.ndarray,
                                  window_size: int = 11) -> float:
    """
    计算半透明模糊度评分（雨滴检测的辅助特征）

    雨滴是半透明的，透过雨滴看到的景物会模糊：
    - 雨滴内部：清晰度降低（拉普拉斯方差小）
    - 雨滴边缘：有折射亮边

    方法：
    1. 计算局部拉普拉斯方差（清晰度）
    2. 计算清晰度的分布偏度
    3. 雨滴区域会有较多"模糊斑块"夹杂在正常清晰度区域中

    Args:
        gray: 灰度图
        window_size: 局部窗口大小
    Returns:
        半透明模糊度评分（归一化到0~1）
    """
    # 计算拉普拉斯
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_abs = np.abs(laplacian)

    # 局部平均拉普拉斯（局部清晰度）
    local_sharpness = cv2.blur(laplacian_abs, (window_size, window_size))

    # 计算清晰度的分布
    # 雨滴区域：有很多低清晰度的小斑块
    # 正常区域：清晰度相对均匀

    # 统计低清晰度像素占比
    blur_threshold = np.median(local_sharpness) * 0.5
    blur_ratio = float(np.sum(local_sharpness < blur_threshold) / local_sharpness.size)

    # 计算清晰度的方差（不均匀程度）
    sharpness_std = float(np.std(local_sharpness))
    sharpness_std_norm = min(1.0, sharpness_std / 20.0)

    # 综合评分
    score = blur_ratio * 0.6 + sharpness_std_norm * 0.4
    return min(1.0, score)


# ============================================================
# 雨滴核心特征 v4：基于边缘的圆形检测
# ============================================================

def detect_rain_circles_edge(gray: np.ndarray,
                              min_radius: int = 2,
                              max_radius: int = 15,
                              circularity_threshold: float = 0.5) -> Tuple[float, int, float]:
    """
    基于边缘的圆形雨滴检测（v4 核心方法）

    核心思想：
    雨滴即使是透明的，边缘也会有折射导致的梯度变化。
    先做边缘检测，再找闭合轮廓，计算圆形度。

    与之前的亮斑法区别：
    - 不依赖亮度阈值，透明雨滴也能检测
    - 直接检测边缘形状，更符合雨滴的物理特征
    - 树叶的边缘是开放的线条，不会被误判为圆形

    步骤：
    1. Canny边缘检测（低阈值，抓弱边缘）
    2. 轮廓检测
    3. 筛选闭合轮廓
    4. 计算每个轮廓的圆形度
    5. 统计近圆形轮廓的密度

    Args:
        gray: 灰度图
        min_radius: 最小半径（像素）
        max_radius: 最大半径（像素）
        circularity_threshold: 圆形度阈值
    Returns:
        (圆形雨滴密度（个/千像素）, 雨滴数量, 平均圆形度)
    """
    h, w = gray.shape

    # 1. Canny边缘检测（v5版阈值）
    median_val = np.median(gray)
    lower = int(max(0, 0.5 * median_val))
    upper = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(gray, lower, upper)

    # 2. 形态学操作
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3. 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    min_area = np.pi * min_radius * min_radius
    max_area = np.pi * max_radius * max_radius

    circle_count = 0
    circularity_sum = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)

        # 面积筛选
        if area < min_area or area > max_area:
            continue

        # 周长
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        # 圆形度：4πA / P²
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity >= circularity_threshold:
            circle_count += 1
            circularity_sum += circularity

    avg_circularity = circularity_sum / circle_count if circle_count > 0 else 0.0
    density = circle_count / (gray.size / 1000.0)

    return density, circle_count, avg_circularity


def calc_rain_edge_density(gray: np.ndarray,
                            min_radius: int = 2,
                            max_radius: int = 15) -> float:
    """
    计算雨滴边缘密度（辅助特征）
    小尺度边缘密度高，大尺度边缘密度低 → 可能是雨滴

    方法：
    1. 计算小尺度边缘（细边缘）
    2. 计算大尺度边缘（粗边缘）
    3. 雨滴特征：小尺度边缘多，大尺度边缘少
    """
    # 小尺度边缘
    edges_small = cv2.Canny(gray, 30, 60)
    small_density = float(np.sum(edges_small > 0) / edges_small.size)

    # 大尺度边缘（先模糊再检测，相当于大尺度）
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges_large = cv2.Canny(blurred, 20, 50)
    large_density = float(np.sum(edges_large > 0) / edges_large.size)

    # 雨滴比值：小尺度边缘多，大尺度边缘少
    if large_density < 0.001:
        ratio = 10.0
    else:
        ratio = small_density / large_density

    # 归一化
    score = min(1.0, ratio / 5.0)
    return score


# ============================================================
# 泥污核心特征：底部区域连通域分析
# ============================================================

def detect_mud_connected_regions(image: np.ndarray,
                                  bottom_ratio: float = 0.25,
                                  min_area_ratio: float = 0.01) -> Tuple[float, List[Tuple[int, int, int, int]]]:
    """
    检测底部区域的泥污连通域（泥污检测的核心方法）

    泥污的关键特征：
    1. 位置：几乎只在画面底部（飞溅的泥水）
    2. 形状：大面积连续的污渍区域
    3. 颜色：深褐、黑褐、黄褐色

    方法：
    1. 只取底部 bottom_ratio 高度的区域
    2. 扩大颜色范围（深褐、黑褐都算）
    3. 形态学闭运算（填充小孔，连接相邻区域）
    4. 连通域分析，只保留大面积的
    5. 返回泥污面积占比和连通域列表

    Args:
        image: BGR图像
        bottom_ratio: 只检测底部多少比例的区域（0~1）
        min_area_ratio: 最小连通域面积占比（相对于底部区域）
    Returns:
        (泥污面积占比（相对于底部区域）, 连通域bbox列表)
    """
    h, w = image.shape[:2]

    # 1. 只取底部区域
    bottom_y = int(h * (1 - bottom_ratio))
    bottom_region = image[bottom_y:h, :, :]
    bottom_h = h - bottom_y
    bottom_area = bottom_h * w

    if bottom_h < 10:
        return 0.0, []

    # 2. 颜色阈值（扩大范围：深褐、黑褐、黄褐、暗灰都算）
    hsv = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)

    # 颜色范围1：黄褐色（H 10~30, S 20~150, V 30~180）
    lower1 = np.array([5, 10, 20])
    upper1 = np.array([35, 180, 200])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # 颜色范围2：深棕色/黑褐色（H 0~20, S 5~150, V 5~120）
    lower2 = np.array([0, 5, 5])
    upper2 = np.array([25, 150, 130])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # 颜色范围3：暗灰色（老旧泥污/水渍）
    lower3 = np.array([0, 0, 5])
    upper3 = np.array([180, 40, 100])
    mask3 = cv2.inRange(hsv, lower3, upper3)

    # 合并掩码
    mud_mask = cv2.bitwise_or(mask1, mask2)
    mud_mask = cv2.bitwise_or(mud_mask, mask3)

    # 3. 形态学操作
    # 闭运算：填充小孔，连接相邻区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mud_mask = cv2.morphologyEx(mud_mask, cv2.MORPH_CLOSE, kernel)

    # 开运算：去除小噪点
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mud_mask = cv2.morphologyEx(mud_mask, cv2.MORPH_OPEN, kernel_small)

    # 4. 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mud_mask, connectivity=8)

    min_area = int(bottom_area * min_area_ratio)

    mud_bboxes = []
    total_mud_area = 0

    for i in range(1, num_labels):  # 跳过背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            ww = stats[i, cv2.CC_STAT_WIDTH]
            hh = stats[i, cv2.CC_STAT_HEIGHT]
            # 转换为原图坐标
            bbox = (x, y + bottom_y, x + ww, y + bottom_y + hh)
            mud_bboxes.append(bbox)
            total_mud_area += area

    mud_ratio = total_mud_area / bottom_area if bottom_area > 0 else 0.0

    return mud_ratio, mud_bboxes


def calc_mud_in_grid(mud_bboxes: List[Tuple[int, int, int, int]],
                     grid_bbox: Tuple[int, int, int, int]) -> float:
    """
    计算某个网格内的泥污面积占比
    Args:
        mud_bboxes: 泥污连通域bbox列表（原图坐标）
        grid_bbox: 网格bbox
    Returns:
        网格内泥污面积占比（0~1）
    """
    gx1, gy1, gx2, gy2 = grid_bbox
    grid_area = (gx2 - gx1) * (gy2 - gy1)
    if grid_area == 0:
        return 0.0

    mud_area_in_grid = 0

    for (mx1, my1, mx2, my2) in mud_bboxes:
        # 计算交集
        ix1 = max(gx1, mx1)
        iy1 = max(gy1, my1)
        ix2 = min(gx2, mx2)
        iy2 = min(gy2, my2)

        if ix2 > ix1 and iy2 > iy1:
            # 简化：用交集面积近似（实际应该用像素级精确计算）
            intersect_area = (ix2 - ix1) * (iy2 - iy1)
            # 乘以一个系数（因为连通域内部不是100%都是泥污）
            mud_area_in_grid += intersect_area * 0.7  # 经验系数

    ratio = min(1.0, mud_area_in_grid / grid_area)
    return ratio


# ============================================================
# 正常场景区域过滤
# ============================================================

def is_normal_scene_region(gray: np.ndarray,
                            color_image: np.ndarray,
                            region_type: str = "auto") -> Tuple[bool, str]:
    """
    判断区域是否是正常场景（天空/道路/植被/建筑）
    用于过滤误判：正常景物不应该被标记为退化

    Args:
        gray: 灰度图
        color_image: 彩色图
        region_type: 强制指定类型，或auto自动判断
    Returns:
        (是否是正常场景, 场景类型)
    """
    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    edge_density = calc_edge_density(gray)

    if len(color_image.shape) == 3:
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        h_mean = float(np.mean(hsv[:, :, 0]))
        s_mean = float(np.mean(hsv[:, :, 1]))
        v_mean = float(np.mean(hsv[:, :, 2]))
    else:
        h_mean = s_mean = v_mean = 0

    # 判断天空：亮度高 + 饱和度低 + 边缘密度极低
    if brightness_mean > 150 and s_mean < 60 and edge_density < 0.02:
        return True, "sky"

    # 判断植被（绿色）：色相在绿色范围 + 饱和度中等 + 边缘密度中等
    if 35 < h_mean < 85 and s_mean > 30 and edge_density > 0.03:
        return True, "vegetation"

    # 判断道路：亮度中等偏低 + 饱和度低 + 边缘密度中等（车道线）
    if 60 < brightness_mean < 130 and s_mean < 40 and 0.01 < edge_density < 0.06:
        return True, "road"

    # 判断建筑：边缘密度高 + 纹理规则 + 亮度多样
    if edge_density > 0.05 and brightness_std > 40:
        return True, "building"

    return False, "unknown"


# ============================================================
# 网格划分与提取
# ============================================================

def divide_grid(image_shape: Tuple[int, int],
                grid_rows: int,
                grid_cols: int) -> List[Tuple[int, int, int, int]]:
    """
    划分图像为网格，返回每个网格的 bbox (x1, y1, x2, y2)
    Args:
        image_shape: (height, width)
        grid_rows: 行数
        grid_cols: 列数
    Returns:
        bbox 列表，按行优先顺序排列
    """
    h, w = image_shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    bboxes = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = (col + 1) * cell_w if col < grid_cols - 1 else w
            y2 = (row + 1) * cell_h if row < grid_rows - 1 else h
            bboxes.append((x1, y1, x2, y2))

    return bboxes


def extract_grid_cell(image: np.ndarray,
                      bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """提取图像中的网格区域"""
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


# ============================================================
# 颜色统计（泥污检测用）
# ============================================================

def calc_color_stats(image: np.ndarray) -> Dict[str, float]:
    """
    计算颜色统计特征（HSV空间）
    Returns:
        包含 h_mean, s_mean, v_mean, s_std, v_std 等的字典
    """
    if len(image.shape) == 2:
        return {"h_mean": 0, "s_mean": 0, "v_mean": float(np.mean(image))}

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    return {
        "h_mean": float(np.mean(h)),
        "s_mean": float(np.mean(s)),
        "v_mean": float(np.mean(v)),
        "s_std": float(np.std(s)),
        "v_std": float(np.std(v)),
    }


# ============================================================
# 可视化
# ============================================================

# 退化类型对应的颜色 (BGR)
DEGRADATION_COLORS = {
    DegradationType.OVEREXPOSURE: (0, 255, 255),   # 黄色 - 过曝
    DegradationType.LOW_LIGHT: (255, 0, 0),         # 蓝色 - 低光
    DegradationType.RAIN_DROP: (255, 255, 0),       # 青色 - 雨滴
    DegradationType.MUD: (0, 140, 255),             # 橙色 - 泥污
    DegradationType.MOTION_BLUR: (128, 0, 128),     # 紫色 - 模糊
    DegradationType.UNKNOWN: (0, 0, 255),           # 红色 - 未知
}


def draw_blind_zones(image: np.ndarray,
                     blind_zones: List[BlindZoneInfo],
                     color_map: Dict = None) -> np.ndarray:
    """
    在图像上绘制盲区标记
    """
    vis = image.copy()
    color_map = color_map or DEGRADATION_COLORS

    for zone in blind_zones:
        color = color_map.get(zone.degradation_type, (0, 0, 255))
        x1, y1, x2, y2 = zone.bbox

        # 半透明填充
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        alpha = 0.3 + zone.confidence * 0.3
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

        # 边框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 标签
        label = f"{zone.degradation_type.value}({zone.severity.value})"
        label = f"{label} {zone.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def draw_grid_debug(image: np.ndarray,
                    regions: List[RegionDegradationInfo],
                    grid_rows: int,
                    grid_cols: int) -> np.ndarray:
    """
    绘制网格质量评分热图（调试用）
    绿色=正常，红色=退化，颜色深浅表示程度
    """
    vis = image.copy()

    for region in regions:
        x1, y1, x2, y2 = region.bbox
        score = region.quality_score

        # 颜色：绿(1.0) -> 黄(0.5) -> 红(0.0)
        if score > 0.5:
            r = int(255 * (1 - score) * 2)
            g = 255
        else:
            r = 255
            g = int(255 * score * 2)
        b = 0

        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (b, g, r), -1)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        # 网格线
        cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # 评分文字
        text = f"{score:.2f}"
        cv2.putText(vis, text, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis
