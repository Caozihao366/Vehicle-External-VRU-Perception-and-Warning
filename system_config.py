"""
车外环视感知VRU轨迹预测与异常行为预警系统 — 统一配置
=====================================================
单路前视视频模式（BEV 退化为 2D 地面平面投影）。
各模块原有 config.py 保持不动，这里只放整合层需要的配置。
"""
from pathlib import Path

# ============================================================
# 路径
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent

VRU_DIR = ROOT_DIR / "vru识别与姿态判定"
FR05_DIR = ROOT_DIR / "fr05_wrongway"
SENSOR_QUALITY_DIR = ROOT_DIR / "sensor_quality"
TTC_DIR = ROOT_DIR / "ttc"
QCNET_DIR = ROOT_DIR / "trajectronpp_ego_bev"
BEV_DIR = ROOT_DIR / "bev_undistort_pipeline3"

# 模型权重
VRU_DET_WEIGHTS = VRU_DIR / "yolov8s.pt"
VRU_POSE_WEIGHTS = VRU_DIR / "yolov8s-pose.pt"
FR05_ORIENT_WEIGHTS = FR05_DIR / "weights" / "vehicle_orientation.pt"
FR05_YOLOPV2_WEIGHTS = FR05_DIR / "third_party" / "YOLOPv2" / "data" / "weights" / "yolopv2.pt"
QCNET_CHECKPOINT = QCNET_DIR / "checkpoints" / "QCNet_Ego_2s_best.ckpt"

# 中文字体（Windows 自带）
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"

# ============================================================
# 输入源
# ============================================================
# 视频路径；0 = 摄像头。可在 main.py 命令行覆盖。
DEFAULT_VIDEO = str(ROOT_DIR / "input" / "test.mp4")
CAMERA_ID = 0

# ============================================================
# 检测类别（统一检测器，以 vru 模块为主）
# ============================================================
# 0=pedestrian, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
DET_CLASSES = [0, 1, 2, 3, 5, 7]
DET_CONF = 0.3
DET_DEVICE = 0          # GPU; 改为 "cpu" 可强制 CPU
DET_HALF = True

# vru 行为分析只关心 VRU 类别
VRU_BEHAVIOR_CLASSES = {0, 1, 3}
# fr05 逆行只关心机动车类别
VEHICLE_CLASSES = {2, 5, 7}

# 类别名（COCO）
CLASS_NAMES = {
    0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck",
}
# 中文类别名
CLASS_NAMES_CN = {
    0: "行人", 1: "自行车", 2: "汽车", 3: "摩托车",
    5: "公交车", 7: "卡车",
}

# ============================================================
# 单目地面平面投影（单路前视 2D 模式）
# ============================================================
# 假设相机安装参数（可根据实际视频标定调整）
CAMERA_HEIGHT_M = 1.5        # 相机离地高度 (m)
CAMERA_PITCH_DEG = 5.0       # 相机俯角 (度)，正值=向下
CAMERA_HFOV_DEG = 70.0       # 水平视场角 (度)
# 垂直视场角由图像宽高比自动推算
MAX_PROJECTION_DIST_M = 50.0 # 最大投影距离 (m)
MIN_PROJECTION_DIST_M = 1.0  # 最近投影距离 (m)

# ============================================================
# 自车运动（单路模式无 CAN，使用配置值）
# ============================================================
EGO_SPEED_MPS = 0.0          # 自车速度 (m/s)，0=静止；可按视频设定
EGO_YAW_RATE_RPS = 0.0       # 自车横摆角速度 (rad/s)
EGO_LENGTH_M = 4.8
EGO_WIDTH_M = 2.0

# ============================================================
# QCNet 轨迹预测
# ============================================================
QCNET_DEVICE = "cuda"
QCNET_FREQUENCY_HZ = 10.0
QCNET_HISTORY_STEPS = 50     # 5 秒历史
QCNET_HORIZON_S = 2.0
# 历史不足时使用匀速模型
QCNET_MIN_HISTORY = 10       # 至少 1 秒历史才启用 QCNet

# agent_type 映射（QCNet AGENT_TYPES 索引）
# 0=vehicle,1=pedestrian,2=motorcyclist,3=cyclist,4=bus
AGENT_TYPE_MAP = {
    0: 1,   # pedestrian -> pedestrian
    1: 3,   # bicycle -> cyclist
    2: 0,   # car -> vehicle
    3: 2,   # motorcycle -> motorcyclist
    5: 4,   # bus -> bus
    7: 0,   # truck -> vehicle
}

# ============================================================
# TTC / 风险
# ============================================================
TTC_HORIZON_S = 5.0
TTC_DT_S = 0.1
TTC_WARNING_S = 2.5
TTC_DANGER_S = 1.2
TTC_SAFETY_MARGIN_M = 0.4
VRU_RADIUS_M = 0.4
RISK_DOWNGRADE_FRAMES = 3
RISK_DANGER_HOLD_FRAMES = 5

# ============================================================
# 传感器质量
# ============================================================
ENABLE_SENSOR_QUALITY = True
# 质量折扣低于此值的检测框将被过滤
MIN_DETECTION_CONFIDENCE = 0.15

# ============================================================
# 逆行检测
# ============================================================
ENABLE_WRONG_WAY = True
ENABLE_YOLOPV2 = True        # 车道线分割（较重，可关）
YOLOPV2_EVERY_N = 5
ORIENT_EVERY_N = 2

# ============================================================
# UI 显示
# ============================================================
UI_WINDOW_NAME = "车外环视VRU感知系统"
UI_FONT_SIZE_SMALL = 16
UI_FONT_SIZE_NORMAL = 20
UI_FONT_SIZE_LARGE = 28
UI_PANEL_WIDTH = 320         # 左侧信息栏宽度
UI_BEV_WIDTH = 480           # 右侧 BEV 视图宽度
UI_TARGET_FPS = 30

# 风险等级颜色 (BGR)
RISK_COLORS = {
    "safe": (0, 200, 0),
    "notice": (0, 200, 200),
    "warning": (0, 165, 255),
    "danger": (0, 0, 255),
}
RISK_NAMES_CN = {
    "safe": "安全",
    "notice": "注意",
    "warning": "警告",
    "danger": "危险",
}
