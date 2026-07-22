import cv2
import numpy as np
# ==================== 通用开关 ====================
FIXED_CAMERA = True          # 固定相机测试视频开，实时摄像头关
# ==================== YOLO与类别 ====================
YOLO_MODEL_PATH = 'yolov8s-pose.pt'
TARGET_CLASSES = [0]
CLASS_NAMES = {0: "pedestrian", 1: "bicycle", 3: "motorcycle"}
CONF_THRESHOLD = 0.3
# ==================== ORB特征匹配 ====================
ORB_NFEATURES = 600
ORB_FAST_THRESHOLD = 20
MIN_MATCH_COUNT = 30
LOWE_RATIO = 0.75
RANSAC_THRESH = 3.0
# ==================== 卡尔曼滤波 ====================
KF_PROCESS_NOISE = {
    'static':      [0.01, 0.01, 0.0001, 0.01, 0.01, 0.0001],
    'constant':    [0.05, 0.05, 0.0005, 0.3,  0.3,  0.002],
    'accelerating':[0.05, 0.05, 0.0005, 1.2,  1.2,  0.01],
    'decelerating':[0.05, 0.05, 0.0005, 0.8,  0.8,  0.006],
}
KF_MEASURE_NOISE = {
    'static':      [0.40, 0.40, 0.002],
    'constant':    [0.15, 0.15, 0.0005],
    'accelerating':[0.10, 0.10, 0.0003],
    'decelerating':[0.20, 0.20, 0.001],
}
MOTION_HISTORY_LEN = 10
STATIC_DEADZONE_PX = 1.0
STATIC_DEADZONE_ANGLE = 0.01
# ==================== 目标跟踪与补偿 ====================
SPEED_WINDOW_SIZE = 8
BBOX_SMOOTH_ALPHA_LARGE = 0.5
BBOX_SMOOTH_ALPHA_SMALL = 0.2
SMALL_BBOX_H_THRESH = 60
LARGE_BBOX_W_THRESH = 100
# 归一化速度阈值（速度算准后阈值）
NORM_STATIC_THRESHOLD = 0.15
NORM_MOVE_THRESHOLD = 0.25
LARGE_MOVE_THRESH_MULT = 1.1
# 状态机确认帧数
STATIC_CONFIRM_FRAMES = 2
MOVE_CONFIRM_FRAMES = 1
SMALL_MOVE_CONFIRM_EXTRA = 0
LARGE_MOVE_CONFIRM_EXTRA = 0
# 噪声与残差
RESIDUAL_SIGMA_MULT = 3.0
DETECTION_JITTER_PX = 2.5
# 局部背景补偿
MIN_RING_PX = 45
MIN_LOCAL_PTS = 6
LOCAL_RING_INIT = 1.5
LOCAL_RING_MAX = 3.0
# 漏检容忍
MAX_LOST_FRAMES = 10
EXTRAPOLATION_DECAY = 0.92
# ==================== 多目标跟踪参数 ====================
TRACK_NEW_CONFIRM_FRAMES = 2
TRACK_MAX_LOST_FRAMES = 15
TRACK_LOST_EXTRAPOLATE = 5
IOU_MATCH_THRESH = 0.3
CENTER_DIST_THRESH = 80
SMALL_IOU_RELAX = 0.2
LARGE_IOU_TIGHTEN = 0.1
TRACK_STATE_NEW = 0
TRACK_STATE_STABLE = 1
TRACK_STATE_LOST = 2
OBJ_KF_PROCESS_NOISE = 0.1
OBJ_KF_MEASURE_NOISE = 1.0
# ==================== 异常行为识别参数 ====================
RUN_ACCEL_THRESH = 0.4
RUN_SPEED_THRESH = 0.8
RUN_CONFIRM_FRAMES = 3
# 跌倒识别（提前触发）
FALL_TORSO_ANGLE = 40
FALL_HW_RATIO = 1.6
FALL_VERTICAL_SPEED = 1.0
FALL_CONFIRM_FRAMES = 1
# 逆行预留
WRONG_WAY_ANGLE_THRESH = 150
WRONG_WAY_CONFIRM_FRAMES = 30
# ==================== 显示 ====================
WINDOW_NAME = "VRU异常行为检测"