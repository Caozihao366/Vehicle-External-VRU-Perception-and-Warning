import cv2
from config import *

def draw_results(frame, results, motion_state, camera_moving, dx, dy, angle_deg, residual, count):
    """绘制所有标注和信息栏，返回绘制后的帧"""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 画每个目标
    for obj in results:
        x1, y1, x2, y2 = obj['bbox']
        state = obj['state']
        track_id = obj['track_id']
        speed = obj['speed_norm']
        thresh = obj['static_thresh']
        used_local = 'L' if obj['used_local'] else 'G'
        bx, by = obj['bottom_center']

        track_state = obj['track_state']
        if track_state == TRACK_STATE_NEW:
            box_color = (0, 255, 255)  # 新生轨迹黄色
        elif track_state == TRACK_STATE_LOST:
            box_color = (128, 128, 128) # 遮挡灰色
        else:
            box_color = (0, 255, 0) if obj['state'] == "static" else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(annotated, (int(bx), int(by)), 4, (255, 255, 0), -1)
        
        label = f"ID:{track_id} {state} v:{speed:.2f}/{thresh:.2f}{used_local}"
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # 左上角信息栏
    info_y = 30
    cam_text = (f"State:{motion_state} | Cam:{'MOVING' if camera_moving else 'STATIC'} "
                f"dx={dx:.1f} dy={dy:.1f} angle={angle_deg:.1f}° res={residual:.1f}px")
    cv2.putText(annotated, cam_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    info_y += 30
    
    for name, num in count.items():
        cv2.putText(annotated, f"{name}: {num}", (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        info_y += 30
    
    cv2.putText(annotated, "Green: static | Red: moving  (L=local, G=global)",
                (20, min(info_y, h-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated