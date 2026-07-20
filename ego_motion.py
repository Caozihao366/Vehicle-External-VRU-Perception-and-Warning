import cv2
import numpy as np
from collections import deque
from config import *

class EgoMotionEstimator:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=ORB_NFEATURES, fastThreshold=ORB_FAST_THRESHOLD)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # 卡尔曼滤波器初始化
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1],
                                              [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], np.float32)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0
        
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.motion_history = deque(maxlen=MOTION_HISTORY_LEN)
        self.motion_state = 'constant'
        self.bg_residual_px = 1.5
        self.bg_pts_prev = None
        self.bg_pts_curr = None

    def _get_motion_state(self):
        """根据位移历史判断运动状态：静止/匀速/加速/减速"""
        if len(self.motion_history) < 5:
            return 'constant'
        recent = list(self.motion_history)[-5:]
        if all(abs(x) < STATIC_DEADZONE_PX for x in recent):
            return 'static'
        diffs = np.diff(recent)
        if np.all(diffs > 0.1):
            return 'accelerating'
        if np.all(diffs < -0.1):
            return 'decelerating'
        return 'constant'

    def _apply_adaptive_kf(self, state):
        """自适应调整卡尔曼噪声参数"""
        self.kf.processNoiseCov = np.diag(KF_PROCESS_NOISE[state]).astype(np.float32)
        self.kf.measurementNoiseCov = np.diag(KF_MEASURE_NOISE[state]).astype(np.float32)

    def update(self, gray, bg_mask, dt):
        """输入当前帧灰度图和前景掩码，输出平滑后的仿射矩阵和状态"""
        camera_moving = False
        measured_dx, measured_dy, measured_angle = 0.0, 0.0, 0.0
        sim_transform = None
        M_smooth = np.eye(2, 3, dtype=np.float32)

        kp, des = self.orb.detectAndCompute(gray, mask=bg_mask)

        if self.prev_gray is not None and des is not None and self.prev_des is not None and len(kp) > 0:
            matches = self.bf.knnMatch(self.prev_des, des, k=2)
            good = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < LOWE_RATIO * n.distance:
                        good.append(m)

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 2)
                sim_transform, mask_inliers = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD
                )
                if sim_transform is not None:
                    inlier_mask = mask_inliers.ravel() == 1
                    self.bg_pts_prev = src_pts[inlier_mask]
                    self.bg_pts_curr = dst_pts[inlier_mask]
                    
                    measured_dx = float(sim_transform[0, 2])
                    measured_dy = float(sim_transform[1, 2])
                    measured_angle = float(np.arctan2(sim_transform[1, 0], sim_transform[0, 0]))
                    self.motion_history.append(measured_dx)

                    # 计算背景残差
                    if len(self.bg_pts_prev) > 0:
                        proj = self.bg_pts_prev @ sim_transform[:, :2].T + sim_transform[:, 2]
                        residuals = np.linalg.norm(proj - self.bg_pts_curr, axis=1)
                        self.bg_residual_px = max(float(np.median(residuals)), 1.0)

                    # 死区判断
                    if abs(measured_dx) > STATIC_DEADZONE_PX or abs(measured_dy) > STATIC_DEADZONE_PX or abs(measured_angle) > STATIC_DEADZONE_ANGLE:
                        camera_moving = True
                    else:
                        measured_dx, measured_dy, measured_angle = 0.0, 0.0, 0.0

        # 自适应卡尔曼滤波
        self.motion_state = self._get_motion_state()
        self._apply_adaptive_kf(self.motion_state)
        self.kf.predict()
        
        if camera_moving:
            measurement = np.array([[measured_dx], [measured_dy], [measured_angle]], dtype=np.float32)
            self.kf.correct(measurement)
        else:
            measurement = np.zeros((3, 1), dtype=np.float32)
            self.kf.correct(measurement)
            if self.motion_state == 'static':
                self.kf.statePost[3:] *= 0.85

        # 重构平滑后的仿射矩阵
        smoothed_dx = self.kf.statePost[0, 0]
        smoothed_dy = self.kf.statePost[1, 0]
        smoothed_angle = self.kf.statePost[2, 0]
        cos_a, sin_a = np.cos(smoothed_angle), np.sin(smoothed_angle)
        M_smooth = np.array([[cos_a, -sin_a, smoothed_dx], [sin_a, cos_a, smoothed_dy]], dtype=np.float32)

        # 更新上一帧状态
        self.prev_gray = gray.copy()
        self.prev_kp, self.prev_des = kp, des

        return M_smooth, camera_moving, smoothed_dx, smoothed_dy, smoothed_angle