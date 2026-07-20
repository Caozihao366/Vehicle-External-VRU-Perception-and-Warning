import numpy as np
from collections import deque
from config import *

class VRUTracker:
    
    def __init__(self):
        self.track_history = {}  # key: track_id → 状态元组
    
    def create_obj_kf(init_x, init_y, init_w, init_h):
        """创建单个目标的卡尔曼滤波器，状态：[x, y, w, h, vx, vy]"""
        kf = cv2.KalmanFilter(6, 4)
        kf.measurementMatrix = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0]
        ], np.float32)
        kf.transitionMatrix = np.array([
            [1,0,0,0,1,0],
            [0,1,0,0,0,1],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ], np.float32)
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * OBJ_KF_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * OBJ_KF_MEASURE_NOISE
        kf.statePost = np.array([[init_x], [init_y], [init_w], [init_h], [0], [0]], np.float32)
        return kf
    
    def _calc_local_bg_disp(self, box_cx, box_cy, bbox_w, bbox_h, bg_pts_prev, bg_pts_curr):
        """计算目标周围局部背景的平均位移"""
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

    def update(self, boxes_data, dt, M_smooth, bg_pts_prev, bg_pts_curr, bg_residual_px):
        """更新所有目标的状态，返回目标结果列表"""
        current_tracks = set()
        results = []

        for (x1, y1, x2, y2, track_id, cls) in boxes_data:
            current_tracks.add(track_id)
            raw_bx = (x1 + x2) // 2
            raw_by = y2
            bbox_w, bbox_h = x2 - x1, y2 - y1
            is_large = bbox_w > LARGE_BBOX_W_THRESH
            is_small = bbox_h < SMALL_BBOX_H_THRESH

            # 坐标平滑
            smooth_alpha = BBOX_SMOOTH_ALPHA_SMALL if is_small else BBOX_SMOOTH_ALPHA_LARGE
            if track_id in self.track_history:
                track = self.track_history[track_id]
                prev_bx = track['smooth_bx']
                prev_by = track['smooth_by']
                speed_window = track['speed_window']
                state = track['motion_state']
                static_cnt = track['static_cnt']
                move_cnt = track['move_cnt']
                lost_cnt = track['lost_cnt']
                last_vx = track['last_vx']
                last_vy = track['last_vy']
                if lost_cnt > 0:
                    smooth_alpha = min(0.6, smooth_alpha * 1.5)
                smooth_bx = smooth_alpha * raw_bx + (1 - smooth_alpha) * prev_bx
                smooth_by = smooth_alpha * raw_by + (1 - smooth_alpha) * prev_by
                lost_cnt = 0
            # 原来的else分支（新目标）替换成：
            else:
            # 新目标第一次出现，初始化轨迹字典
                raw_cx = (x1 + x2) / 2
                raw_cy = (y1 + y2) / 2
                self.track_history[track_id] = {
                    'track_id': track_id,
                    'cls': cls,
                    'state': TRACK_STATE_NEW,
                    'kf': create_obj_kf(raw_cx, raw_cy, bbox_w, bbox_h),
                    'hit_cnt': 1,
                    'lost_cnt': 0,
                    'bbox': (x1, y1, x2, y2),
                    'bottom_center': (raw_bx, raw_by),
                    'smooth_bx': raw_bx,
                    'smooth_by': raw_by,
                    'speed_window': deque(maxlen=SPEED_WINDOW_SIZE),
                    'speed_norm': 0,
                    'static_thresh': NORM_STATIC_THRESHOLD,
                    'used_local': False,
                    'motion_state': 'static',
                    'static_cnt': 0,
                    'move_cnt': 0,
                    'last_vx': 0.0,
                    'last_vy': 0.0
                }
                instant_speed_norm = 0
                bg_consistent = False
                used_local = False

            # 运动补偿计算速度
            instant_speed_norm = 0
            bg_consistent = False
            used_local = False

            if track_id in self.track_history:
                box_cx, box_cy = (x1 + x2) // 2, (y1 + y2) // 2
                local_dx, local_dy = self._calc_local_bg_disp(box_cx, box_cy, bbox_w, bbox_h, bg_pts_prev, bg_pts_curr)
                
                if local_dx is not None:
                    used_local = True
                    exp_bx = prev_bx + local_dx
                    exp_by = prev_by + local_dy
                    rel_dx = (smooth_bx - prev_bx) - local_dx
                    rel_dy = (smooth_by - prev_by) - local_dy
                    local_rel_speed = np.sqrt(rel_dx**2 + rel_dy**2) / dt
                    local_rel_speed_norm = local_rel_speed / max(bbox_h, 20)
                    if local_rel_speed_norm < NORM_STATIC_THRESHOLD:
                        bg_consistent = True
                else:
                    P_prev = np.array([[prev_bx], [prev_by], [1.0]], dtype=np.float32)
                    P_exp = M_smooth @ P_prev
                    exp_bx, exp_by = float(P_exp[0, 0]), float(P_exp[1, 0])

                real_dx = smooth_bx - exp_bx
                real_dy = smooth_by - exp_by
                comp_speed_pixels = np.sqrt(real_dx**2 + real_dy**2) / dt
                instant_speed_norm = comp_speed_pixels / max(bbox_h, 20)
                last_vx, last_vy = real_dx / dt, real_dy / dt
                speed_window.append(instant_speed_norm)

            avg_speed_norm = np.mean(speed_window) if len(speed_window) > 0 else 0

            # 自适应阈值
            pixel_noise_floor = RESIDUAL_SIGMA_MULT * bg_residual_px + DETECTION_JITTER_PX
            noise_floor_norm = (pixel_noise_floor / dt) / max(bbox_h, 20)
            adaptive_static_thresh = max(NORM_STATIC_THRESHOLD, noise_floor_norm)
            adaptive_move_thresh = max(NORM_MOVE_THRESHOLD, noise_floor_norm * 2.0)

            # 大目标特殊阈值
            move_confirm_needed = MOVE_CONFIRM_FRAMES
            if is_large:
                adaptive_move_thresh *= LARGE_MOVE_THRESH_MULT
                move_confirm_needed += LARGE_MOVE_CONFIRM_EXTRA
            if is_small:
                move_confirm_needed += SMALL_MOVE_CONFIRM_EXTRA

            # 状态机
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

            track['smooth_bx'] = smooth_bx
            track['smooth_by'] = smooth_by
            track['speed_window'] = speed_window
            track['motion_state'] = state
            track['static_cnt'] = static_cnt
            track['move_cnt'] = move_cnt
            track['lost_cnt'] = 0
            track['last_vx'] = last_vx
            track['last_vy'] = last_vy
            track['bbox'] = (x1, y1, x2, y2)
            track['bottom_center'] = (smooth_bx, smooth_by)
            track['speed_norm'] = avg_speed_norm
            track['static_thresh'] = adaptive_static_thresh
            track['used_local'] = used_local
            track['hit_cnt'] += 1
            # 连续3帧检测到，新生轨迹转稳定
            if track['state'] == TRACK_STATE_NEW and track['hit_cnt'] >= TRACK_NEW_CONFIRM_FRAMES:
                track['state'] = TRACK_STATE_STABLE
            
            results.append({
                'track_id': track_id,
                'cls': cls,
                'bbox': (x1, y1, x2, y2),
                'bottom_center': (smooth_bx, smooth_by),
                'state': state,
                'speed_norm': avg_speed_norm,
                'static_thresh': adaptive_static_thresh,
                'used_local': used_local
            })

        # 漏检外推
        for tid in list(self.track_history.keys()):
            if tid not in current_tracks:
                track = self.track_history[tid]
                track['lost_cnt'] += 1
                if track['lost_cnt'] >= TRACK_MAX_LOST_FRAMES:
                    del self.track_history[tid]
                else:
                    # 前5帧做指数衰减外推
                    if track['lost_cnt'] <= TRACK_LOST_EXTRAPOLATE:
                        decay = EXTRAPOLATION_DECAY ** track['lost_cnt']
                        track['last_vx'] *= decay
                        track['last_vy'] *= decay
                        track['smooth_bx'] += track['last_vx'] * dt
                        track['smooth_by'] += track['last_vy'] * dt
                        track['state'] = TRACK_STATE_LOST
                    self.track_history[tid] = track