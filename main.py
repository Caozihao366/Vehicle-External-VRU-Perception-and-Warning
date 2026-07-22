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
    model = YOLO(YOLO_MODEL_PATH)
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

        # 1. YOLO检测跟踪（pose模型输出和原来完全兼容，多了keypoints）
        results = model.track(
            frame, device=0, classes=[0], conf=CONF_THRESHOLD,
            tracker="bytetrack.yaml", verbose=False, persist=True, half=True
        )

        # 2. 生成背景掩码 + 取检测框+关键点
        bg_mask = np.ones(gray.shape, dtype=np.uint8) * 255
        boxes_data = []
        keypoints_data = {}
        count = {"pedestrian": 0}

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clses = results[0].boxes.cls.cpu().numpy().astype(int)
            # 取关键点：拼接xy和置信度，变成[N,17,3] (x,y,conf)
            if results[0].keypoints is not None:
                kp_xy = results[0].keypoints.xy.cpu().numpy()
                kp_conf = results[0].keypoints.conf.cpu().numpy()
                # 把置信度拼到最后一维
                keypoints = np.concatenate([kp_xy, kp_conf[..., None]], axis=-1)
            else:
                keypoints = None

            for i, (box, track_id, cls) in enumerate(zip(boxes, ids, clses)):
                x1, y1, x2, y2 = map(int, box)
                boxes_data.append((x1, y1, x2, y2, track_id, cls))
                count["pedestrian"] += 1
                # 存关键点
                if keypoints is not None and i < len(keypoints):
                    keypoints_data[track_id] = keypoints[i]
                # 掩码外扩15像素，剔除目标边缘特征
                mx1, my1 = max(0, x1-15), max(0, y1-15)
                mx2, my2 = min(w, x2+15), min(h, y2+15)
                bg_mask[my1:my2, mx1:mx2] = 0

        # 3. 自运动估计
        M_smooth, camera_moving, dx, dy, angle_rad = ego_estimator.update(gray, bg_mask, dt)
        angle_deg = angle_rad * 180 / np.pi

        # 4. VRU目标跟踪与运动补偿（传入关键点）
        vru_results = tracker.update(
            boxes_data, keypoints_data, dt, M_smooth,
            ego_estimator.bg_pts_prev, ego_estimator.bg_pts_curr,
            ego_estimator.bg_residual_px
        )

        # 5. 行为分析
        for obj in vru_results:
            behavior, conf = behavior_analyzer.analyze(obj, dt, h, w)
            obj['behavior'] = behavior
            obj['behavior_conf'] = conf

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