from config import *
import numpy as np
from collections import deque

class BehaviorAnalyzer:
    def __init__(self):
        self.behavior_history = {}

    def _analyze_visible_skeleton(self, kps):
        """
        【泛化遮挡处理核心】
        分析可见关键点的综合主轴倾角与最高稳定锚点。
        无论画面中剩下的是上半身、左半身还是只有一条腿，都能计算出其实际的物理姿态倾向。
        """
        stable_y = None
        skeleton_angle = None
        
        if kps is not None and len(kps) >= 17:
            # 1. 寻找最顶端可靠关键点 (优先级: 鼻子 > 肩膀 > 骨盆)
            # 用于追踪绝对的重力下落速度，不受 BBox 变形影响
            for i in [0, 5, 6, 11, 12]:
                if kps[i][2] > 0.4:
                    stable_y = kps[i][1]
                    break
                    
            # 2. 动态提取“可见子图 (Visible Sub-graph)”的主轴
            def get_mid(kp1, kp2):
                if kp1[2] > 0.3 and kp2[2] > 0.3:
                    return ((kp1[0]+kp2[0])/2, (kp1[1]+kp2[1])/2)
                elif kp1[2] > 0.3: return (kp1[0], kp1[1])
                elif kp2[2] > 0.3: return (kp2[0], kp2[1])
                return None
                
            nose = kps[0]
            shoulder_mid = get_mid(kps[5], kps[6])
            hip_mid = get_mid(kps[11], kps[12])
            l_knee, r_knee = kps[13], kps[14]
            l_ankle, r_ankle = kps[15], kps[16]
            
            # 收集当前帧中所有置信度及格的躯干向量
            vectors = []
            if nose[2] > 0.3 and shoulder_mid:
                vectors.append((shoulder_mid[0]-nose[0], shoulder_mid[1]-nose[1]))
            if shoulder_mid and hip_mid:
                vectors.append((hip_mid[0]-shoulder_mid[0], hip_mid[1]-shoulder_mid[1]))
            if kps[11][2] > 0.3 and l_knee[2] > 0.3:
                vectors.append((l_knee[0]-kps[11][0], l_knee[1]-kps[11][1]))
            if kps[12][2] > 0.3 and r_knee[2] > 0.3:
                vectors.append((r_knee[0]-kps[12][0], r_knee[1]-kps[12][1]))
            if l_knee[2] > 0.3 and l_ankle[2] > 0.3:
                vectors.append((l_ankle[0]-l_knee[0], l_ankle[1]-l_knee[1]))
            if r_knee[2] > 0.3 and r_ankle[2] > 0.3:
                vectors.append((r_ankle[0]-r_knee[0], r_ankle[1]-r_knee[1]))
                
            angles = []
            for v in vectors:
                norm = np.linalg.norm(v)
                if norm > 5: # 过滤极短向量引起的角度噪声
                    v_norm = v / norm
                    # 计算与绝对垂直重力方向 (0, 1) 的夹角
                    cos_theta = np.dot(v_norm, np.array([0, 1.0]))
                    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    angle = min(angle, 180 - angle) # 将角度折叠到 [0, 90] 度，只看倾斜度
                    angles.append(angle)
            
            # 3. 计算所有可见部件的平均方向
            if angles:
                skeleton_angle = np.mean(angles)
                
        return stable_y, skeleton_angle

    def analyze(self, track, dt, frame_h=None, frame_w=None):
        track_id = track['track_id']
        if track_id not in self.behavior_history:
            self.behavior_history[track_id] = {
                'running_cnt': 0,
                'falling_cnt': 0,
                'fall_triggered': False,
                'recover_cnt': 0,
                'speed_hist': deque(maxlen=30),
                'hw_ratio_hist': deque(maxlen=30),
                'stable_y_hist': deque(maxlen=30)
            }
        hist = self.behavior_history[track_id]
        x1, y1, x2, y2 = track['bbox']
        bbox_h = track['bbox_h']
        bbox_w = x2 - x1
        hw_ratio = bbox_h / max(bbox_w, 1)
        kps = track.get('keypoints', None)
        hist['speed_hist'].append(track['speed_norm'])
        hist['hw_ratio_hist'].append(hw_ratio)
        behavior = 'normal'
        confidence = 0.0

        # 获取泛化姿态倾角与最高稳定锚点
        stable_y, skeleton_angle = self._analyze_visible_skeleton(kps)
        if stable_y is not None:
            # 过滤关键点跳变：stable_y和上一帧差超过bbox高度的一半，说明换了关键点，不存
            if len(hist['stable_y_hist']) == 0 or abs(stable_y - hist['stable_y_hist'][-1]) < bbox_h * 0.5:
                hist['stable_y_hist'].append(stable_y)

        # 边缘截断检测：bbox贴画面边缘时，高宽比不可信，屏蔽bbox判断
        is_edge_truncated = False
        if frame_h is not None and frame_w is not None:
            edge_margin = 10
            if x1 < edge_margin or y1 < edge_margin or x2 > frame_w - edge_margin or y2 > frame_h - edge_margin:
                is_edge_truncated = True

        # ==================== 跌倒状态保持与解除 ====================
        if hist['fall_triggered']:
            behavior = 'falling'
            confidence = 0.9
            # 解除条件：连续3帧可见骨架直立，才解除，避免跳变
            is_upright = (skeleton_angle is not None and skeleton_angle < 40)
            if is_upright:
                hist['recover_cnt'] +=1
            else:
                hist['recover_cnt'] = max(0, hist['recover_cnt']-1)
            if hist['recover_cnt'] >=3:
                hist['fall_triggered'] = False
                hist['falling_cnt'] = 0
                hist['recover_cnt'] = 0

        # ==================== 跌倒识别（最高优先级，提前触发）====================
        if not hist['fall_triggered'] and len(hist['stable_y_hist']) >=10: # 新目标前10帧不判，避免初始化误判
            fall_score = 0
            
            # 计算绝对重心下落速度 (必要条件)
            vertical_speed = 0
            if len(hist['stable_y_hist']) >= 10:
                dy = hist['stable_y_hist'][-1] - hist['stable_y_hist'][-10]
                vertical_speed = (dy / (9*dt)) / max(bbox_h, 20)
            has_falling_velocity = vertical_speed > FALL_VERTICAL_SPEED

            # bbox高宽比变化
            hw_ratio_drop = 0
            history_len = len(hist['hw_ratio_hist'])
            if history_len >= 10:
                hw_ratio_drop = hist['hw_ratio_hist'][-10] - hw_ratio
            ratio_low = hw_ratio < FALL_HW_RATIO
            ratio_fast_drop = hw_ratio_drop > 0.3
            
            # 1. 骨架特征判定（最高优先级，抗遮挡）
            if skeleton_angle is not None:
                if skeleton_angle > FALL_TORSO_ANGLE:
                    if has_falling_velocity:
                        fall_score += 2  # 姿态水平+高速下落=完美跌倒特征
                    else:
                        fall_score += 1  # 姿态水平但无速度（已躺平/爬行）
                elif skeleton_angle < 45:
                    # 一票否决：可见骨架直立，不管bbox怎么变都不判跌倒
                    pass

            # 2. 骨架失效时，退化为bbox+速度判断（边缘截断时屏蔽bbox）
            else:
                if not is_edge_truncated:
                    if hw_ratio < 1.0:
                        fall_score += 2 if has_falling_velocity else 1
                    elif ratio_low: 
                        fall_score += 1
                    if ratio_fast_drop and has_falling_velocity: 
                        fall_score += 1

            # 盲区跌倒
            blind_fall = track['lost_cnt'] > 3 and (ratio_low or (skeleton_angle is not None and skeleton_angle > FALL_TORSO_ANGLE))
            if blind_fall: fall_score += 2

            # 分数结算
            if fall_score >= 2:
                hist['falling_cnt'] += 1
            else:
                hist['falling_cnt'] = max(0, hist['falling_cnt']-1)
            
            if hist['falling_cnt'] >= FALL_CONFIRM_FRAMES:
                behavior = 'falling'
                confidence = 0.9
                hist['fall_triggered'] = True
                hist['running_cnt'] = 0

        # ==================== 奔跑识别（跌倒/下落时直接抑制）====================
        if behavior == 'normal' and len(hist['speed_hist']) >= 10:
            speeds = list(hist['speed_hist'])
            accel = (speeds[-1] - speeds[-5]) / (4*dt)
            current_speed = np.mean(speeds[-5:])
            vertical_speed_ratio = abs(track['last_vy']) / max(abs(track['last_vx']), 0.1)
            
            hw_ratio_drop_recent = 0
            if len(hist['hw_ratio_hist']) >=10:
                hw_ratio_drop_recent = hist['hw_ratio_hist'][-10] - hw_ratio
            # 有下落速度/高宽比快速下降时，直接抑制奔跑
            if (accel > RUN_ACCEL_THRESH and current_speed > RUN_SPEED_THRESH 
                and vertical_speed_ratio < 0.6 and hw_ratio_drop_recent < 0.2
                and vertical_speed < FALL_VERTICAL_SPEED):
                hist['running_cnt'] += 1
            else:
                hist['running_cnt'] = max(0, hist['running_cnt']-1)
            if hist['running_cnt'] >= RUN_CONFIRM_FRAMES:
                behavior = 'running'
                confidence = min(0.9, hist['running_cnt']/10)

        # 清理丢失目标
        if track['track_state'] == TRACK_STATE_LOST and track['lost_cnt'] > TRACK_MAX_LOST_FRAMES-2:
            if track_id in self.behavior_history:
                del self.behavior_history[track_id]

        return behavior, confidence