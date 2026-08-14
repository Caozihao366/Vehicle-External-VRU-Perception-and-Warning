"""
VRU 感知模块包装器
==================
将 "vru识别与姿态判定" 目录下的脚本式代码封装为可导入的类。
该模块原有文件使用扁平导入 (from config import *)，
因此这里通过 sys.path 注入其目录后再导入。
原有文件不做任何修改。
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---- 注入 vru 模块目录 ----
_VRU_DIR = Path(__file__).resolve().parent / "vru识别与姿态判定"
if str(_VRU_DIR) not in sys.path:
    sys.path.insert(0, str(_VRU_DIR))

# 先导入 config 并覆盖需要整合层控制的参数，再导入其他子模块
import config as _vru_cfg

# 整合层：车载相机随自车运动，需要背景补偿
_vru_cfg.FIXED_CAMERA = False

from ego_motion import EgoMotionEstimator      # noqa: E402
from vru_tracker import VRUTracker             # noqa: E402
from behavior_analyzer import BehaviorAnalyzer  # noqa: E402

# 清理 sys.path（避免污染其他模块导入）
sys.path.remove(str(_VRU_DIR))


class VRUPerception:
    """
    VRU 检测 + 跟踪 + 姿态 + 异常行为（奔跑/跌倒）。

    用法:
        perc = VRUPerception()
        tracks = perc.process_frame(frame_bgr, dt)
    """

    def __init__(
        self,
        det_weights: str = None,
        pose_weights: str = None,
        det_classes: list = None,
        device=0,
        conf: float = 0.3,
        half: bool = True,
    ):
        det_weights = det_weights or str(_VRU_DIR / "yolov8s.pt")
        pose_weights = pose_weights or str(_VRU_DIR / "yolov8s-pose.pt")

        self.device = device
        self.conf = conf
        self.half = half
        self.det_classes = det_classes if det_classes is not None else [0, 1, 2, 3, 5, 7]

        self.model_det = YOLO(det_weights)
        self.model_pose = YOLO(pose_weights)

        self.ego_estimator = EgoMotionEstimator()
        self.tracker = VRUTracker()
        self.behavior_analyzer = BehaviorAnalyzer()

        self._prev_time = None

    def process_frame(self, frame: np.ndarray, dt: float = None) -> dict:
        """
        处理一帧。

        返回 dict:
            tracks: list[dict]  — 每个目标的跟踪+行为信息
            ego_motion: dict    — 相机自运动信息
            count: dict         — 各类型计数
        """
        if dt is None:
            now = time.time()
            dt = max(now - self._prev_time, 0.001) if self._prev_time else 0.033
            self._prev_time = now

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- YOLO 检测 + ByteTrack ----
        results = self.model_det.track(
            frame, device=self.device, classes=self.det_classes,
            conf=self.conf, tracker="bytetrack.yaml",
            verbose=False, persist=True, half=self.half,
        )

        bg_mask = np.ones(gray.shape, dtype=np.uint8) * 255
        boxes_data = []
        keypoints_data = {}
        count = {}

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clses = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, clses):
                x1, y1, x2, y2 = map(int, box)
                boxes_data.append((x1, y1, x2, y2, track_id, cls))
                count[_vru_cfg.CLASS_NAMES.get(cls, str(cls))] = \
                    count.get(_vru_cfg.CLASS_NAMES.get(cls, str(cls)), 0) + 1

                # 背景掩码：排除目标区域
                mx1, my1 = max(0, x1 - 15), max(0, y1 - 15)
                mx2, my2 = min(w, x2 + 15), min(h, y2 + 15)
                bg_mask[my1:my2, mx1:mx2] = 0

                # 仅对行人跑姿态估计
                if cls == 0:
                    pad = 20
                    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
                    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
                    roi = frame[ry1:ry2, rx1:rx2]
                    if roi.shape[0] > 20 and roi.shape[1] > 20:
                        pose_res = self.model_pose(
                            roi, verbose=False, half=self.half
                        )[0]
                        if pose_res.keypoints is not None and len(pose_res.keypoints.xy) > 0:
                            kp_xy = pose_res.keypoints.xy.cpu().numpy()[0]
                            kp_conf = pose_res.keypoints.conf.cpu().numpy()[0]
                            kp_xy[:, 0] += rx1
                            kp_xy[:, 1] += ry1
                            keypoints_data[track_id] = np.concatenate(
                                [kp_xy, kp_conf[..., None]], axis=-1
                            )

        # ---- 自运动估计 ----
        M_smooth, camera_moving, dx, dy, angle_rad = self.ego_estimator.update(
            gray, bg_mask, dt
        )
        angle_deg = angle_rad * 180 / np.pi

        # ---- VRU 跟踪 + 运动补偿 ----
        tracks = self.tracker.update(
            boxes_data, keypoints_data, dt, M_smooth,
            self.ego_estimator.bg_pts_prev,
            self.ego_estimator.bg_pts_curr,
            self.ego_estimator.bg_residual_px,
        )

        # ---- 人车绑定 ----
        BehaviorAnalyzer.bind_rider_to_vehicle(tracks)

        # ---- 异常行为分析（仅 VRU 类） ----
        for obj in tracks:
            if obj.get("is_rider", False):
                obj["behavior"] = "normal"
                obj["behavior_conf"] = 1.0
                continue
            if obj["cls"] in (0, 1, 3):
                behavior, conf = self.behavior_analyzer.analyze(obj, dt, h, w)
                obj["behavior"] = behavior
                obj["behavior_conf"] = conf
            else:
                obj["behavior"] = "normal"
                obj["behavior_conf"] = 1.0

        # 跌倒记忆池（ghost targets）
        tracks.extend(self.behavior_analyzer.get_active_ghosts())

        ego_motion = {
            "M_smooth": M_smooth,
            "camera_moving": camera_moving,
            "dx": dx,
            "dy": dy,
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
            "motion_state": self.ego_estimator.motion_state,
            "bg_residual_px": self.ego_estimator.bg_residual_px,
        }

        return {"tracks": tracks, "ego_motion": ego_motion, "count": count}
