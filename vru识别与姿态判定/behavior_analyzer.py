from config import *
import numpy as np
from collections import deque
import time

class BehaviorAnalyzer:
    def __init__(self):
        self.behavior_history = {}
        self.ghost_falls = {}  # 丢失跌倒/侧翻目标记忆池
        self.GHOST_MAX_AGE = 450

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

        # ====== 基于类别的分流逻辑，绝不干扰原有的行人(cls=0)判别 ======
        cls = track.get('cls', 0)
        if cls in [1, 3]:  # 1: 自行车, 3: 摩托车
            return self._analyze_twowheeler(track, dt, hist, hw_ratio, skeleton_angle, is_edge_truncated)

        # ==================== 跌倒状态保持与解除 ====================
        if hist['fall_triggered']:
            behavior = 'falling'
            confidence = 0.9
            if track.get('lost_cnt', 0) > 0:
                pass
            else:
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
            blind_fall = track.get('lost_cnt', 0) > 3 and (ratio_low or (skeleton_angle is not None and skeleton_angle > FALL_TORSO_ANGLE))
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
            vertical_speed_ratio = abs(track.get('last_vy', 0)) / max(abs(track.get('last_vx', 0.1)), 0.1)
            
            hw_ratio_drop_recent = 0
            if len(hist['hw_ratio_hist']) >=10:
                hw_ratio_drop_recent = hist['hw_ratio_hist'][-10] - hw_ratio
            # 有下落速度/高宽比快速下降时，直接抑制奔跑
            vertical_speed = 0
            if len(hist['stable_y_hist']) >= 10:
                dy = hist['stable_y_hist'][-1] - hist['stable_y_hist'][-10]
                vertical_speed = (dy / (9*dt)) / max(bbox_h, 20)
                
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
        if track.get('track_state') == TRACK_STATE_LOST and track.get('lost_cnt', 0) > TRACK_MAX_LOST_FRAMES-2:
            # 行人跌倒删除前存入记忆池
            if hist.get('fall_triggered', False):
                self.ghost_falls[track_id] = {
                    'bbox': track['bbox'],
                    'cls': track.get('cls', 0),
                    'age': 0,
                    'behavior': 'falling'
                }
            if track_id in self.behavior_history:
                del self.behavior_history[track_id]

        return behavior, confidence

    def _analyze_twowheeler(self, track, dt, hist, hw_ratio, skeleton_angle, is_edge_truncated=False):
        """
        专用于二轮车（自行车/摩托车）的异常行为识别逻辑。
        底层复用运动学特征，上层阈值与判定树完全解耦。
        """
        behavior = 'normal'
        confidence = 0.0
        bbox_h = track['bbox_h']
        cls = track.get('cls', 1)
        is_lost = (track.get('track_state') == TRACK_STATE_LOST)

        # ==================== 1. 侧倒(Falling)状态保持与解除 ====================
        if hist['fall_triggered']:
            behavior = 'falling'
            confidence = 0.9
            if track.get('lost_cnt', 0) > 0:
                pass
            else:
                # 解除条件：连续3帧宽高比恢复高瘦(>1.2)，或绑定的骑手骨架恢复直立(<30度)
                is_upright = (skeleton_angle is not None and skeleton_angle < 30)
                if hw_ratio > 1.2 or is_upright:
                    hist['recover_cnt'] += 1
                else:
                    hist['recover_cnt'] = max(0, hist['recover_cnt'] - 1)
                
                if hist['recover_cnt'] >= 3:
                    hist['fall_triggered'] = False
                    hist['falling_cnt'] = 0
                    hist['recover_cnt'] = 0

        # ==================== 2. 侧倒识别触发逻辑 ====================
        if not hist['fall_triggered'] and len(hist['stable_y_hist']) >= 10:
            fall_score = 0
            
            # 计算绝对重心/最高锚点下落速度 (必要条件)
            vertical_speed = 0
            if len(hist['stable_y_hist']) >= 10:
                dy = hist['stable_y_hist'][-1] - hist['stable_y_hist'][-10]
                vertical_speed = (dy / (9*dt)) / max(bbox_h, 20)
                
            has_falling_velocity = vertical_speed > FALL_VERTICAL_SPEED
            
            # 二轮车独有的 Bbox 塌陷判断 (注意：这里使用从 config 导入的常量)
            ratio_collapsed = hw_ratio < BIKE_FALL_HW_RATIO
            ratio_fast_drop = False
            if len(hist['hw_ratio_hist']) >= 10:
                ratio_fast_drop = (hist['hw_ratio_hist'][-10] - hw_ratio) > 0.3

            # 融合判定：骨骼信息(微观) + 检测框信息(宏观)
            if skeleton_angle is not None:
                if skeleton_angle > BIKE_FALL_TORSO_ANGLE:
                    fall_score += 2 if has_falling_velocity else 1
                elif skeleton_angle < 30:
                    # 【一票否决】姿态直立：只要骑手没倒，车框再怎么变也不算跌倒
                    pass
            else:
                # 兜底：无骨骼点时，纯靠 BBox 和速度（边缘截断时屏蔽 BBox）
                if not is_edge_truncated:
                    if ratio_collapsed:
                        fall_score += 2 if has_falling_velocity else 1
                    if ratio_fast_drop and has_falling_velocity:
                        fall_score += 1

            if fall_score >= 2:
                hist['falling_cnt'] += 1
            else:
                hist['falling_cnt'] = max(0, hist['falling_cnt'] - 1)
                
            if hist['falling_cnt'] >= FALL_CONFIRM_FRAMES:
                behavior = 'falling'
                confidence = 0.9
                hist['fall_triggered'] = True
                hist['running_cnt'] = 0

        # ==================== 3. 突然加速(Sudden Accel)识别 ====================
        if behavior == 'normal' and len(hist['speed_hist']) >= 10:
            speeds = list(hist['speed_hist'])
            accel = (speeds[-1] - speeds[-5]) / (4*dt)
            current_speed = np.mean(speeds[-5:])
            
            vertical_speed_ratio = abs(track.get('last_vy', 0)) / max(abs(track.get('last_vx', 0.1)), 0.1)
            
            hw_ratio_drop_recent = 0
            if len(hist['hw_ratio_hist']) >= 10:
                hw_ratio_drop_recent = hist['hw_ratio_hist'][-10] - hw_ratio
                
            vertical_speed = 0
            if len(hist['stable_y_hist']) >= 10:
                dy = hist['stable_y_hist'][-1] - hist['stable_y_hist'][-10]
                vertical_speed = (dy / (9*dt)) / max(bbox_h, 20)
            
            # 根据具体车型动态选择阈值 (使用从 config 导入的常量)
            accel_thresh = MOTO_ACCEL_THRESH if cls == 3 else BIKE_ACCEL_THRESH
            speed_thresh = MOTO_SPEED_THRESH if cls == 3 else BIKE_SPEED_THRESH
            
            if (accel > accel_thresh and current_speed > speed_thresh 
                and vertical_speed_ratio < 0.6 and hw_ratio_drop_recent < 0.2
                and vertical_speed < FALL_VERTICAL_SPEED):
                hist['running_cnt'] += 1
            else:
                hist['running_cnt'] = max(0, hist['running_cnt'] - 1)
                
            if hist['running_cnt'] >= RUN_CONFIRM_FRAMES:
                # 状态机复用 running，业务层对应二轮车的“突然加速”
                behavior = 'running' 
                confidence = min(0.9, hist['running_cnt']/10)
                
        # 同步清理丢失目标
                # 同步清理丢失目标
        if track.get('track_state') == TRACK_STATE_LOST and track.get('lost_cnt', 0) > TRACK_MAX_LOST_FRAMES-2:
            track_id = track['track_id']
            # 跌倒/侧翻目标删除前存入记忆池
            if hist.get('fall_triggered', False):
                self.ghost_falls[track_id] = {
                    'bbox': track['bbox'],
                    'cls': track.get('cls', 1),
                    'age': 0,
                    'behavior': 'falling'
                }
            if track_id in self.behavior_history:
                del self.behavior_history[track_id]
        return behavior, confidence
    
    @staticmethod
    def bind_rider_to_vehicle(current_tracks):
        """
        前置核心动作：人车实体绑定 (Entity Binding)
        将画面中的骑手姿态挂载到对应的二轮车 track 字典上。
        应在每一帧 Tracker 更新完成后，调用 analyze 之前执行此方法。
        
        :param current_tracks: list[dict], 包含当前帧所有有效目标的状态字典
        """
        persons = [t for t in current_tracks if t.get('cls') == 0 and t.get('keypoints') is not None]
        vehicles = [t for t in current_tracks if t.get('cls') in [1, 3]]
        
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v['bbox']
            best_person = None
            max_ioa = 0.0  # 行人面积被包含的比例 (Intersection over Person Area)
            
            for p in persons:
                px1, py1, px2, py2 = p['bbox']
                
                # 计算交集
                ix1 = max(vx1, px1)
                iy1 = max(vy1, py1)
                ix2 = min(vx2, px2)
                iy2 = min(vy2, py2)
                
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    p_area = (px2 - px1) * (py2 - py1)
                    ioa = inter_area / p_area if p_area > 0 else 0
                    
                    # 强关联判定：如果行人的大部分面积(>40%)在车框内
                    if ioa > max_ioa and ioa > 0.4:
                        max_ioa = ioa
                        best_person = p
                        
            if best_person is not None:
                # 将人的关键点指针赋予车辆，形成骑行者联合体 (Rider-Vehicle Unit)
                v['keypoints'] = best_person['keypoints']
                v['stable_y_hist'] = best_person.get('stable_y_hist', deque(maxlen=30))
                # 打上被绑定标签，可用于外层跳过骑手的单独判定
                best_person['is_rider'] = True

    def get_active_ghosts(self):
        """返回存活的记忆池报警目标，自动超时删除"""
        active = []
        to_del = []
        for gid, ghost in self.ghost_falls.items():
            ghost['age'] += 1
            if ghost['age'] > self.GHOST_MAX_AGE:
                to_del.append(gid)
            else:
                x1,y1,x2,y2 = ghost['bbox']
                active.append({
                        'track_id': gid,
                        'cls': ghost['cls'],
                        'track_state': TRACK_STATE_LOST,
                        'bbox': ghost['bbox'],
                        'bottom_center': ((x1+x2)//2, y2),
                        'state': 'static',
                        'speed_norm': 0,
                        'static_thresh': 0,
                        'used_local': False,
                        'keypoints': None,
                        'lost_cnt': 999,
                        'last_vx':0, 'last_vy':0,
                        'bbox_h': y2-y1,
                        'behavior': ghost['behavior'],
                        'is_ghost': True
                })
        for gid in to_del:
            del self.ghost_falls[gid]
        return active