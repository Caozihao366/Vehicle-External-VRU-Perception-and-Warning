"""
传感器质量检测系统 - 使用示例

演示：
1. 基本使用流程
2. 单帧图像检测
3. 视频/连续帧检测
4. 可视化调试
5. 与下游模块（检测/跟踪）的联动方式
"""

import cv2
import numpy as np
import os
import sys

# 确保可以导入包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensor_quality import (
    SensorQualityManager,
    SensorQualityConfig,
    GlobalCheckerConfig,
    RegionalDiagnoserConfig,
    DegradationDecisionConfig,
    DegradationType,
    GlobalQualityStatus,
)


# ============================================================
# 示例1：基本使用 - 单帧检测
# ============================================================

def demo_single_frame(image_path: str):
    """
    示例1：单帧图像质量检测
    """
    print("=" * 60)
    print("示例1：单帧图像质量检测")
    print("=" * 60)

    # 1. 创建管理器（使用默认配置）
    config = SensorQualityConfig(
        enable_regional_diagnosis=True,
        enable_temporal_smoothing=True,
        enable_profiling=True,
    )
    manager = SensorQualityManager(config)

    # 2. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    print(f"图像尺寸: {image.shape}")

    # 3. 处理（前几帧用于时序缓冲，所以处理5帧模拟连续输入）
    for i in range(5):
        result = manager.process_frame(image, frame_id=i + 1)

    # 4. 输出结果
    print(f"\n全局状态: {result.global_status.value}")
    print(f"全局质量评分: {result.global_quality_score:.3f}")
    print(f"有效视野比例: {result.effective_fov_ratio:.1%}")
    print(f"检测置信度折扣: {result.detection_confidence_discount:.1%}")
    print(f"跟踪可靠性: {result.tracking_reliability:.1%}")
    print(f"系统建议: {result.suggestion}")

    print(f"\n检测到 {len(result.blind_zones)} 个盲区:")
    for i, zone in enumerate(result.blind_zones):
        print(f"  盲区{i+1}: 类型={zone.degradation_type.value}, "
              f"严重度={zone.severity.value}, "
              f"位置={zone.bbox}, "
              f"置信度={zone.confidence:.2f}")

    # 5. 性能统计
    stats = manager.get_profiling_stats()
    if stats:
        print(f"\n性能统计:")
        for key, val in stats.items():
            print(f"  {key}: avg={val['avg_ms']:.2f}ms, "
                  f"max={val['max_ms']:.2f}ms, count={val['count']}")

    # 6. 可视化并保存
    vis = manager.visualize(image, result, mode="both")
    output_path = "demo_single_frame_result.jpg"
    cv2.imwrite(output_path, vis)
    print(f"\n可视化结果已保存: {output_path}")

    return result


# ============================================================
# 示例2：模拟不同退化场景
# ============================================================

def demo_degradation_scenarios():
    """
    示例2：模拟各种退化场景，验证检测效果
    """
    print("\n" + "=" * 60)
    print("示例2：模拟各种退化场景")
    print("=" * 60)

    # 创建一个测试图像（模拟正常道路场景）
    h, w = 720, 1280
    base_image = np.zeros((h, w, 3), dtype=np.uint8)

    # 画一个简单的道路场景
    # 天空
    base_image[:240, :] = [200, 180, 150]  # BGR
    # 道路
    base_image[240:, :] = [80, 80, 80]
    # 车道线
    for x in range(100, w, 200):
        cv2.rectangle(base_image, (x, 400), (x + 20, 450), (255, 255, 255), -1)
    # 一些边缘纹理（增加清晰度）
    cv2.line(base_image, (0, 240), (w, 240), (100, 100, 100), 2)

    manager = SensorQualityManager()

    scenarios = [
        ("正常图像", base_image.copy()),
    ]

    # 场景1：局部过曝（右上角）
    overexposed = base_image.copy()
    overexposed[50:200, 900:1200] = [255, 255, 255]  # 过曝区域
    scenarios.append(("局部过曝", overexposed))

    # 场景2：整体低光
    lowlight = base_image.copy()
    lowlight = (lowlight * 0.2).astype(np.uint8)
    scenarios.append(("整体低光", lowlight))

    # 场景3：局部泥污（左下角）
    muddy = base_image.copy()
    # 模拟泥污：低对比度、低饱和度、偏黄褐色
    mud_region = np.zeros((200, 300, 3), dtype=np.uint8)
    mud_region[:] = [60, 80, 100]  # 黄褐色
    # 加一点噪声
    noise = np.random.randint(0, 10, mud_region.shape, dtype=np.uint8)
    mud_region = cv2.add(mud_region, noise)
    muddy[520:720, 0:300] = mud_region
    scenarios.append(("局部泥污", muddy))

    # 场景4：雨滴（随机亮斑）
    rainy = base_image.copy()
    # 随机生成雨滴亮斑
    num_drops = 200
    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        size = np.random.randint(2, 8)
        cv2.circle(rainy, (x, y), size, (255, 255, 255), -1)
    scenarios.append(("雨滴", rainy))

    # 场景5：运动模糊
    blurred = base_image.copy()
    kernel_size = 25
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(blurred, -1, kernel)
    scenarios.append(("运动模糊", blurred))

    # 测试每个场景
    results = []
    for name, img in scenarios:
        # 处理5帧（时序缓冲）
        manager.reset()
        for i in range(5):
            result = manager.process_frame(img, frame_id=i + 1)

        results.append((name, result))
        print(f"\n【{name}】")
        print(f"  状态: {result.global_status.value}")
        print(f"  质量评分: {result.global_quality_score:.3f}")
        print(f"  盲区数量: {len(result.blind_zones)}")
        if result.blind_zones:
            for z in result.blind_zones[:3]:  # 最多显示3个
                print(f"    - {z.degradation_type.value} ({z.severity.value}) "
                      f"@ {z.bbox}, conf={z.confidence:.2f}")

    # 保存可视化结果
    for name, img in scenarios:
        manager.reset()
        for i in range(5):
            result = manager.process_frame(img, frame_id=i + 1)
        vis = manager.visualize(img, result, mode="both")
        safe_name = name.replace("/", "_")
        cv2.imwrite(f"demo_{safe_name}.jpg", vis)

    print("\n所有场景可视化结果已保存")


# ============================================================
# 示例3：与下游检测模块的联动
# ============================================================

def demo_downstream_integration():
    """
    示例3：演示如何与下游检测/跟踪模块联动
    """
    print("\n" + "=" * 60)
    print("示例3：与下游检测/跟踪模块联动")
    print("=" * 60)

    manager = SensorQualityManager()

    # 模拟一帧有局部退化的图像
    h, w = 720, 1280
    image = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)

    # 加一个过曝区域
    image[100:300, 800:1100] = 255

    # 处理图像
    for i in range(5):
        result = manager.process_frame(image, frame_id=i + 1)

    # 模拟下游检测结果
    detections = [
        {"class": "pedestrian", "bbox": [100, 400, 150, 550], "confidence": 0.92},
        {"class": "car", "bbox": [850, 150, 1000, 280], "confidence": 0.88},  # 在过曝区域
        {"class": "cyclist", "bbox": [500, 350, 550, 480], "confidence": 0.85},
    ]

    print("\n原始检测结果:")
    for det in detections:
        print(f"  {det['class']}: conf={det['confidence']:.2f}, bbox={det['bbox']}")

    # 应用传感器质量折扣
    # 逻辑：如果检测框落入盲区，降低其置信度
    adjusted_detections = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        det_center_x = (x1 + x2) / 2
        det_center_y = (y1 + y2) / 2

        # 检查是否在盲区内
        in_blind_zone = False
        zone_confidence = 1.0
        for zone in result.blind_zones:
            zx1, zy1, zx2, zy2 = zone.bbox
            if zx1 <= det_center_x <= zx2 and zy1 <= det_center_y <= zy2:
                in_blind_zone = True
                zone_confidence = min(zone_confidence, 1.0 - zone.confidence * 0.5)
                break

        # 应用全局折扣 + 局部盲区折扣
        final_confidence = det['confidence'] * result.detection_confidence_discount
        if in_blind_zone:
            final_confidence *= zone_confidence

        adjusted_detections.append({
            **det,
            "adjusted_confidence": final_confidence,
            "in_blind_zone": in_blind_zone,
        })

    print("\n调整后检测结果:")
    for det in adjusted_detections:
        status = "⚠️ 盲区中" if det['in_blind_zone'] else "✓ 正常"
        print(f"  {det['class']}: "
              f"原conf={det['confidence']:.2f} → "
              f"调整后={det['adjusted_confidence']:.2f} "
              f"{status}")

    # 跟踪建议
    print(f"\n跟踪模块建议:")
    print(f"  全局可靠性系数: {result.tracking_reliability:.2f}")
    if result.global_status == GlobalQualityStatus.DEGRADED:
        print("  → 建议：切换到纯运动模型外推模式，减少检测更新频率")
    elif result.global_status == GlobalQualityStatus.PARTIALLY_DEGRADED:
        print("  → 建议：盲区内目标增加历史轨迹权重，延长丢失容忍时间")
    else:
        print("  → 建议：正常跟踪")


# ============================================================
# 示例4：自定义配置
# ============================================================

def demo_custom_config():
    """
    示例4：自定义配置参数
    """
    print("\n" + "=" * 60)
    print("示例4：自定义配置")
    print("=" * 60)

    # 自定义各模块配置
    global_config = GlobalCheckerConfig(
        downsample_max_size=320,           # 更小的降采样，更快
        overexposed_ratio_threshold=0.20,  # 放宽过曝阈值
        blur_laplacian_var_threshold=80.0,  # 放宽模糊阈值
    )

    regional_config = RegionalDiagnoserConfig(
        grid_rows=4,                       # 更少的网格，更快
        grid_cols=6,
        mud_temporal_stable_frames=3,      # 更快确认泥污
    )

    decision_config = DegradationDecisionConfig(
        temporal_window_size=5,            # 更大的时序窗口，更稳定
        temporal_vote_threshold=0.8,        # 更严格的投票阈值
        mild_degradation_ratio=0.15,        # 调整轻度降级阈值
    )

    config = SensorQualityConfig(
        enable_regional_diagnosis=True,
        enable_temporal_smoothing=True,
        global_config=global_config,
        regional_config=regional_config,
        decision_config=decision_config,
    )

    manager = SensorQualityManager(config)
    print("已创建自定义配置的管理器")
    print(f"  全局降采样尺寸: {global_config.downsample_max_size}")
    print(f"  网格划分: {regional_config.grid_rows}x{regional_config.grid_cols}")
    print(f"  时序窗口: {decision_config.temporal_window_size}帧")

    # 简单测试
    test_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    result = manager.process_frame(test_img, frame_id=1)
    print(f"\n测试图像检测结果:")
    print(f"  状态: {result.global_status.value}")
    print(f"  质量评分: {result.global_quality_score:.3f}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("传感器质量检测与降级系统 - 使用示例")
    print("FR-08: Sensor Quality Detection & Degradation")
    print()

    # 示例2：模拟各种退化场景（不需要外部图片）
    demo_degradation_scenarios()

    # 示例3：下游联动
    demo_downstream_integration()

    # 示例4：自定义配置
    demo_custom_config()

    # 如果有测试图片，可以运行示例1
    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        demo_single_frame(test_image_path)
    else:
        print(f"\n提示: 如需运行单帧检测示例，请将图片命名为 {test_image_path} 放在当前目录")

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
