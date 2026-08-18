import cv2
import numpy as np
from collections import deque
from .config import *


def create_obj_kf(init_cx, init_cy, init_w, init_h):
    """6 状态 [cx,cy,w,h,vx,vy]；4 观测 [cx,cy,w,h]。"""
    kf = cv2.KalmanFilter(6, 4)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ], np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * OBJ_KF_PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * OBJ_KF_MEASURE_NOISE
    kf.statePost = np.array(
        [[init_cx], [init_cy], [init_w], [init_h], [0], [0]], np.float32
    )
    return kf


def _kf_set_dt(kf: cv2.KalmanFilter, dt: float):
    dt = max(float(dt), 1e-3)
    kf.transitionMatrix[0, 4] = dt
    kf.transitionMatrix[1, 5] = dt


def _kf_predict_correct(kf: cv2.KalmanFilter, cx, cy, w, h, dt: float):
    _kf_set_dt(kf, dt)
    kf.predict()
    meas = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
    kf.correct(meas)


def _kf_predict_only(kf: cv2.KalmanFilter, dt: float):
    _kf_set_dt(kf, dt)
    kf.predict()


def _kf_read_state(kf: cv2.KalmanFilter):
    """由框中心状态换算底点与速度（px/s，dt 已写入转移矩阵）。"""
    cx = float(kf.statePost[0, 0])
    cy = float(kf.statePost[1, 0])
    bw = max(float(kf.statePost[2, 0]), 1.0)
    bh = max(float(kf.statePost[3, 0]), 1.0)
    vx = float(kf.statePost[4, 0])
    vy = float(kf.statePost[5, 0])
    bx = int(round(cx))
    by = int(round(cy + 0.5 * bh))
    return bx, by, bw, bh, vx, vy


class VRUTracker:
    """基于 ByteTrack ID + 卡尔曼滤波的 VRU 运动状态估计与轨迹维护。"""

    def __init__(self):
        self.track_history = {}

    def _calc_local_bg_disp(self, box_cx, box_cy, bbox_w, bbox_h, bg_pts_prev, bg_pts_curr):
        if bg_pts_prev is None or len(bg_pts_prev) == 0:
            return None, None
        dist_x = np.abs(bg_pts_prev[:, 0] - box_cx)
        dist_y = np.abs(bg_pts_prev[:, 1] - box_cy)
        ring_out_x = max(bbox_w * LOCAL_RING_INIT, MIN_RING_PX * LOCAL_RING_INIT)
        ring_out_y = max(bbox_h * LOCAL_RING_INIT, MIN_RING_PX * LOCAL_RING_INIT)
        expand = LOCAL_RING_INIT
        local_bg_prev, local_bg_curr = [], []
        while len(local_bg_prev) < MIN_LOCAL_PTS and expand <= LOCAL_RING_MAX:
            nearby_mask = (dist_x < ring_out_x * expand) & (dist_y < ring_out_y * expand)
            local_bg_prev = bg_pts_prev[nearby_mask]
            local_bg_curr = bg_pts_curr[nearby_mask]
            expand += 0.5
        if len(local_bg_prev) >= MIN_LOCAL_PTS:
            local_dx = float(np.median(local_bg_curr[:, 0] - local_bg_prev[:, 0]))
            local_dy = float(np.median(local_bg_curr[:, 1] - local_bg_prev[:, 1]))
            return local_dx, local_dy
        return None, None

    @staticmethod
    def _heading_from_velocity(vx, vy, min_speed_px=8.0):
        if vx * vx + vy * vy < min_speed_px * min_speed_px:
            return None
        return float(np.degrees(np.arctan2(-vy, vx)))

    @staticmethod
    def _heading_from_traj(traj):
        if len(traj) < 2:
            return None
        x0, y0 = traj[0]
        x1, y1 = traj[-1]
        dx, dy = x1 - x0, y1 - y1
        if dx * dx + dy * dy < 16.0:
            return None
        return float(np.degrees(np.arctan2(-dy, dx)))

    def update(
        self,
        boxes_data,
        dt,
        M_smooth=None,
        bg_pts_prev=None,
        bg_pts_curr=None,
        bg_residual_px=0.0,
    ):
        if M_smooth is None:
            M_smooth = np.eye(3, dtype=np.float32)

        current_tracks = set()
        results = []

        for (x1, y1, x2, y2, track_id, cls) in boxes_data:
            current_tracks.add(track_id)
            raw_bx = (x1 + x2) // 2
            raw_by = y2
            bbox_w, bbox_h = x2 - x1, y2 - y1
            box_cx = 0.5 * (x1 + x2)
            box_cy = 0.5 * (y1 + y2)
            is_large = bbox_w > LARGE_BBOX_W_THRESH
            is_small = bbox_h < SMALL_BBOX_H_THRESH

            if track_id in self.track_history:
                track = self.track_history[track_id]
                prev_raw_bx = track["raw_bx"]
                prev_raw_by = track["raw_by"]
                speed_window = track["speed_window"]
                state = track["motion_state"]
                static_cnt = track["static_cnt"]
                move_cnt = track["move_cnt"]

                # ---- 卡尔曼：predict + correct ----
                kf = track["kf"]
                _kf_predict_correct(kf, box_cx, box_cy, bbox_w, bbox_h, dt)
                smooth_bx, smooth_by, _, kf_h, last_vx, last_vy = _kf_read_state(kf)

                bg_consistent = False
                used_local = False
                box_cx_i, box_cy_i = (x1 + x2) // 2, (y1 + y2) // 2

                if FIXED_CAMERA:
                    inst_dx = raw_bx - prev_raw_bx
                    inst_dy = raw_by - prev_raw_by
                else:
                    local_dx, local_dy = self._calc_local_bg_disp(
                        box_cx_i, box_cy_i, bbox_w, bbox_h, bg_pts_prev, bg_pts_curr
                    )
                    if local_dx is not None:
                        used_local = True
                        exp_bx = prev_raw_bx + local_dx
                        exp_by = prev_raw_by + local_dy
                        inst_dx = raw_bx - exp_bx
                        inst_dy = raw_by - exp_by
                        local_rel_speed = np.sqrt(inst_dx**2 + inst_dy**2) / dt
                        local_rel_speed_norm = local_rel_speed / max(bbox_h, 20)
                        if local_rel_speed_norm < NORM_STATIC_THRESHOLD:
                            bg_consistent = True
                    else:
                        p_prev = np.array([[prev_raw_bx], [prev_raw_by], [1.0]], dtype=np.float32)
                        p_exp = M_smooth @ p_prev
                        exp_bx, exp_by = float(p_exp[0, 0]), float(p_exp[1, 0])
                        inst_dx = raw_bx - exp_bx
                        inst_dy = raw_by - exp_by

                comp_speed_pixels = np.sqrt(last_vx**2 + last_vy**2)
                instant_speed_norm = comp_speed_pixels / max(bbox_h, 20)

                speed_window.append(instant_speed_norm)
                avg_speed_norm = float(np.mean(speed_window)) if len(speed_window) > 0 else 0.0

                if FIXED_CAMERA:
                    pixel_noise_floor = DETECTION_JITTER_PX * 0.5
                else:
                    pixel_noise_floor = RESIDUAL_SIGMA_MULT * bg_residual_px + DETECTION_JITTER_PX

                noise_floor_norm = (pixel_noise_floor / dt) / max(bbox_h, 20)
                adaptive_static_thresh = max(NORM_STATIC_THRESHOLD, noise_floor_norm)
                adaptive_move_thresh = max(NORM_MOVE_THRESHOLD, noise_floor_norm * 1.5)
                move_confirm_needed = MOVE_CONFIRM_FRAMES
                if is_large:
                    adaptive_move_thresh *= LARGE_MOVE_THRESH_MULT
                    move_confirm_needed += LARGE_MOVE_CONFIRM_EXTRA
                if is_small:
                    move_confirm_needed += SMALL_MOVE_CONFIRM_EXTRA

                if state == "static":
                    if avg_speed_norm > adaptive_move_thresh:
                        move_cnt += 1
                        static_cnt = 0
                        if move_cnt >= move_confirm_needed:
                            state = "moving"
                    else:
                        move_cnt = 0
                else:
                    if avg_speed_norm < adaptive_static_thresh or bg_consistent:
                        static_cnt += 1
                        move_cnt = 0
                        if static_cnt >= STATIC_CONFIRM_FRAMES:
                            state = "static"
                    else:
                        static_cnt = 0

                track["raw_bx"] = raw_bx
                track["raw_by"] = raw_by
                track["smooth_bx"] = smooth_bx
                track["smooth_by"] = smooth_by
                track["speed_window"] = speed_window
                track["motion_state"] = state
                track["static_cnt"] = static_cnt
                track["move_cnt"] = move_cnt
                track["lost_cnt"] = 0
                track["last_vx"] = last_vx
                track["last_vy"] = last_vy
                track["bbox"] = (x1, y1, x2, y2)
                track["bottom_center"] = (smooth_bx, smooth_by)
                track["speed_norm"] = avg_speed_norm
                track["static_thresh"] = adaptive_static_thresh
                track["used_local"] = used_local
                track["hit_cnt"] += 1
                track["trajectory"].append((float(smooth_bx), float(smooth_by)))
                if track["state"] == TRACK_STATE_NEW and track["hit_cnt"] >= TRACK_NEW_CONFIRM_FRAMES:
                    track["state"] = TRACK_STATE_STABLE

                heading = self._heading_from_velocity(last_vx, last_vy)
                if heading is None:
                    heading = self._heading_from_traj(track["trajectory"])
                prev_h = track.get("heading_deg")
                if heading is not None and prev_h is not None:
                    delta = (heading - prev_h + 180) % 360 - 180
                    heading = prev_h + 0.4 * delta
                track["heading_deg"] = heading

                results.append({
                    "track_id": track_id,
                    "cls": cls,
                    "class_name": CLASS_NAMES.get(cls, str(cls)),
                    "track_state": track["state"],
                    "bbox": (x1, y1, x2, y2),
                    "bottom_center": (smooth_bx, smooth_by),
                    "state": state,
                    "speed_norm": avg_speed_norm,
                    "static_thresh": adaptive_static_thresh,
                    "used_local": used_local,
                    "lost_cnt": track["lost_cnt"],
                    "last_vx": last_vx,
                    "last_vy": last_vy,
                    "bbox_h": bbox_h,
                    "trajectory": list(track["trajectory"]),
                    "heading_deg": heading,
                    "behavior": "unknown",
                })
            else:
                kf = create_obj_kf(box_cx, box_cy, bbox_w, bbox_h)
                self.track_history[track_id] = {
                    "track_id": track_id,
                    "cls": cls,
                    "state": TRACK_STATE_NEW,
                    "kf": kf,
                    "hit_cnt": 1,
                    "lost_cnt": 0,
                    "bbox": (x1, y1, x2, y2),
                    "raw_bx": raw_bx,
                    "raw_by": raw_by,
                    "bottom_center": (raw_bx, raw_by),
                    "smooth_bx": raw_bx,
                    "smooth_by": raw_by,
                    "speed_window": deque(maxlen=SPEED_WINDOW_SIZE),
                    "speed_norm": 0.0,
                    "static_thresh": NORM_STATIC_THRESHOLD,
                    "used_local": False,
                    "motion_state": "static",
                    "static_cnt": 0,
                    "move_cnt": 0,
                    "last_vx": 0.0,
                    "last_vy": 0.0,
                    "trajectory": deque([(float(raw_bx), float(raw_by))], maxlen=TRAJ_HISTORY_LEN),
                    "heading_deg": None,
                }
                results.append({
                    "track_id": track_id,
                    "cls": cls,
                    "class_name": CLASS_NAMES.get(cls, str(cls)),
                    "track_state": TRACK_STATE_NEW,
                    "bbox": (x1, y1, x2, y2),
                    "bottom_center": (raw_bx, raw_by),
                    "state": "static",
                    "speed_norm": 0.0,
                    "static_thresh": NORM_STATIC_THRESHOLD,
                    "used_local": False,
                    "lost_cnt": 0,
                    "last_vx": 0.0,
                    "last_vy": 0.0,
                    "bbox_h": bbox_h,
                    "trajectory": [(float(raw_bx), float(raw_by))],
                    "heading_deg": None,
                    "behavior": "unknown",
                })

        for tid in list(self.track_history.keys()):
            if tid not in current_tracks:
                track = self.track_history[tid]
                track["lost_cnt"] += 1
                if track["lost_cnt"] >= TRACK_MAX_LOST_FRAMES:
                    del self.track_history[tid]
                elif track["lost_cnt"] <= TRACK_LOST_EXTRAPOLATE:
                    kf = track["kf"]
                    _kf_predict_only(kf, dt)
                    decay = EXTRAPOLATION_DECAY ** track["lost_cnt"]
                    kf.statePost[4, 0] *= decay
                    kf.statePost[5, 0] *= decay
                    smooth_bx, smooth_by, _, _, last_vx, last_vy = _kf_read_state(kf)
                    track["smooth_bx"] = smooth_bx
                    track["smooth_by"] = smooth_by
                    track["last_vx"] = last_vx
                    track["last_vy"] = last_vy
                    track["bottom_center"] = (smooth_bx, smooth_by)
                    track["state"] = TRACK_STATE_LOST
                    track["trajectory"].append((float(smooth_bx), float(smooth_by)))

        return results
