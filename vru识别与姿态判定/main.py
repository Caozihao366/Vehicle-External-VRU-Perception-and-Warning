import cv2
import time
from ultralytics import YOLO
from config import *
from ego_motion import EgoMotionEstimator
from vru_tracker import VRUTracker
from behavior_analyzer import BehaviorAnalyzer
from visualization import draw_results

FIXED_CAMERA = False

def main():
    # 初始化所有模块
    model_det = YOLO(DET_MODEL_PATH)
    model_pose = YOLO(POSE_MODEL_PATH)
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    ego_estimator = EgoMotionEstimator()
    tracker = VRUTracker()
    behavior_analyzer = BehaviorAnalyzer()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        dt = max(current_time - prev_time, 0.001)
        prev_time = current_time
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # YOLO检测跟踪（pose模型输出和原来完全兼容，多了keypoints）
        # 主模型检测三类目标+跟踪
        results = model_det.track(
            frame, device=0, classes=TARGET_CLASSES, conf=CONF_THRESHOLD,
            tracker="bytetrack.yaml", verbose=False, persist=True,
            half=True
        )

        bg_mask = np.ones(gray.shape, dtype=np.uint8) * 255
        boxes_data = []
        keypoints_data = {}
        count = {}

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clses = results[0].boxes.cls.cpu().numpy().astype(int)

            for i, (box, track_id, cls) in enumerate(zip(boxes, ids, clses)):
                x1, y1, x2, y2 = map(int, box)
                boxes_data.append((x1, y1, x2, y2, track_id, cls))
                count[CLASS_NAMES[cls]] = count.get(CLASS_NAMES[cls], 0) + 1
                mx1, my1 = max(0, x1-15), max(0, y1-15)
                mx2, my2 = min(w, x2+15), min(h, y2+15)
                bg_mask[my1:my2, mx1:mx2] = 0

                # 只有行人跑pose，自行车摩托车跳过
                if cls == 0:
                    pad = 20
                    roi_x1, roi_y1 = max(0, x1-pad), max(0, y1-pad)
                    roi_x2, roi_y2 = min(w, x2+pad), min(h, y2+pad)
                    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    if roi.shape[0]>20 and roi.shape[1]>20:
                        pose_res = model_pose(roi, verbose=False, half=True)[0]
                        if pose_res.keypoints is not None and len(pose_res.keypoints.xy) > 0:
                            kp_xy = pose_res.keypoints.xy.cpu().numpy()[0]
                            kp_conf = pose_res.keypoints.conf.cpu().numpy()[0]
                            # 坐标还原回原图
                            kp_xy[:, 0] += roi_x1
                            kp_xy[:, 1] += roi_y1
                            kps = np.concatenate([kp_xy, kp_conf[..., None]], axis=-1)
                            keypoints_data[track_id] = kps

        # 3. 自运动估计
        M_smooth, camera_moving, dx, dy, angle_rad = ego_estimator.update(gray, bg_mask, dt)
        angle_deg = angle_rad * 180 / np.pi
        bg_residual = ego_estimator.bg_residual_px

        # 4. VRU目标跟踪与运动补偿（传入关键点）
        vru_results = tracker.update(
            boxes_data, keypoints_data, dt, M_smooth,
            ego_estimator.bg_pts_prev, ego_estimator.bg_pts_curr,
            bg_residual
        )

        # 人车绑定
        BehaviorAnalyzer.bind_rider_to_vehicle(vru_results)

        for obj in vru_results:
            if obj.get('is_rider', False):
                obj['behavior'] = 'normal'
                continue
            behavior, conf = behavior_analyzer.analyze(obj, dt, h, w)
            obj['behavior'] = behavior
            obj['behavior_conf'] = conf

        vru_results.extend(behavior_analyzer.get_active_ghosts())

        # 6. 绘制结果
        annotated = draw_results(frame, vru_results, ego_estimator.motion_state,
                                camera_moving, dx, dy, angle_deg, ego_estimator.bg_residual_px, count)
        cv2.imshow(WINDOW_NAME, annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()