"""FR05 逆行检测模块默认配置（路径相对本包目录）。"""
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent

# ==================== 通用开关 ====================
FIXED_CAMERA = True
DEVICE = "auto"              # auto | mps | cuda:0 | cpu
YOLOPV2_DEVICE = None        # None=跟随 DEVICE
ORIENT_DEVICE = None

# ==================== YOLO 与类别 ====================
DET_MODEL_PATH = str(_PKG_ROOT / "weights" / "yolov8s.pt")
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
CLASS_NAMES = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
CONF_THRESHOLD = 0.3
TRACKER_CFG = "bytetrack.yaml"

# ==================== YOLOPv2 ====================
YOLOPV2_WEIGHTS = str(_PKG_ROOT / "third_party" / "YOLOPv2" / "data" / "weights" / "yolopv2.pt")
YOLOPV2_IMG_SIZE = 640
ENABLE_YOLOPV2 = True
YOLOPV2_EVERY_N = 5

# ==================== 目标跟踪与补偿 ====================
SPEED_WINDOW_SIZE = 8
BBOX_SMOOTH_ALPHA_LARGE = 0.5
BBOX_SMOOTH_ALPHA_SMALL = 0.2
SMALL_BBOX_H_THRESH = 60
LARGE_BBOX_W_THRESH = 100
NORM_STATIC_THRESHOLD = 0.15
NORM_MOVE_THRESHOLD = 0.25
LARGE_MOVE_THRESH_MULT = 1.1
STATIC_CONFIRM_FRAMES = 2
MOVE_CONFIRM_FRAMES = 1
SMALL_MOVE_CONFIRM_EXTRA = 0
LARGE_MOVE_CONFIRM_EXTRA = 0
RESIDUAL_SIGMA_MULT = 3.0
DETECTION_JITTER_PX = 2.5
MIN_RING_PX = 45
MIN_LOCAL_PTS = 6
LOCAL_RING_INIT = 1.5
LOCAL_RING_MAX = 3.0
EXTRAPOLATION_DECAY = 0.92

# ==================== 多目标跟踪参数 ====================
TRACK_NEW_CONFIRM_FRAMES = 2
TRACK_MAX_LOST_FRAMES = 15
TRACK_LOST_EXTRAPOLATE = 5
TRACK_STATE_NEW = 0
TRACK_STATE_STABLE = 1
TRACK_STATE_LOST = 2
OBJ_KF_PROCESS_NOISE = 0.1
OBJ_KF_MEASURE_NOISE = 1.0
TRAJ_HISTORY_LEN = 30

# ==================== 车辆朝向 ====================
ORIENT_WEIGHTS = str(_PKG_ROOT / "weights" / "vehicle_orientation.pt")
ORIENT_IMG_SIZE = 224
ORIENT_CONF_THRESH = 0.55
VEHICLE_ORIENT_CLASSES = [2, 3, 5, 7]
ORIENT_EVERY_N = 2
ORIENT_PRIMARY_WEIGHT = 1.0
ORIENT_AUX_WEIGHT = 0.45

# ==================== 本车道 ====================
EGO_INVADE_OVERLAP = 0.08
DRAW_EGO_LANE = True
DRAW_LANE = True

# ==================== 逆行检测 ====================
WRONG_WAY_CONFIRM_FRAMES = 4
WRONG_WAY_CLEAR_FRAMES = 10
WRONG_WAY_SCORE_DECAY = 0.4
WRONG_WAY_ANGLE_THRESH = 160
WRONG_WAY_MIN_SPEED = 0.10

# ==================== 显示（demo 用） ====================
WINDOW_NAME = "FR05 Wrong-way"
DRAW_TRAJECTORY = False
