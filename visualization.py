import cv2
import numpy as np
from config import *

def draw_results(frame, results, motion_state, camera_moving, dx, dy, angle_deg, residual, count):
    h, w = frame.shape[:2]
    panel_w = 280
    # 创建整体画布：左侧信息栏 + 右侧视频
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    # 左侧信息栏：深灰色背景 + 黑色边框
    canvas[:, :panel_w] = (30, 30, 30)
    cv2.rectangle(canvas, (0,0), (panel_w-1, h-1), (0,0,0), 2)
    # 右侧贴视频帧
    canvas[:, panel_w:panel_w+w] = frame

    # ==================== 绘制视频区域的检测框和标签 ====================
    video_offset_x = panel_w
    for obj in results:
        x1, y1, x2, y2 = obj['bbox']
        # 坐标偏移到视频区域
        x1 += video_offset_x
        x2 += video_offset_x
        state = obj['state']
        track_id = obj['track_id']
        speed = obj['speed_norm']
        thresh = obj['static_thresh']
        used_local = 'L' if obj['used_local'] else 'G'
        bx, by = obj['bottom_center']
        bx += video_offset_x
        behavior = obj.get('behavior', 'normal')
        track_state = obj['track_state']

        # 颜色和线宽
        if behavior == 'falling':
            box_color = (0, 0, 255)
            behavior_tag = ' [FALL!]'
            thickness = 3
        elif behavior == 'running':
            box_color = (0, 165, 255)
            behavior_tag = ' [RUN!]'
            thickness = 2
        elif behavior == 'wrong_way':
            box_color = (255, 0, 255)
            behavior_tag = ' [WRONG WAY]'
            thickness = 2
        else:
            if track_state == TRACK_STATE_NEW:
                box_color = (0, 255, 255)
                behavior_tag = ' [NEW]'
                thickness = 1
            elif track_state == TRACK_STATE_LOST:
                box_color = (128, 128, 128)
                behavior_tag = ' [OCC]'
                thickness = 1
            else:
                box_color = (0, 255, 0) if state == "static" else (0, 0, 255)
                behavior_tag = ''
                thickness = 2

        # 画框和底部中心点
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
        cv2.circle(canvas, (int(bx), int(by)), 4, (255, 255, 0), -1)
        
        # 标签文字加黑色描边（黑边）
        label = f"ID:{track_id} {state} v:{speed:.2f}{used_local}{behavior_tag}"
        text_y = y1 - 8
        # 先画4个方向的黑色文字做描边
        cv2.putText(canvas, label, (x1+1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(canvas, label, (x1-1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(canvas, label, (x1, text_y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(canvas, label, (x1, text_y-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        # 再画彩色文字
        cv2.putText(canvas, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # ==================== 绘制左侧信息栏 ====================
    text_x = 20
    cur_y = 40
    # 标题
    cv2.putText(canvas, "VRU PERCEPTION", (text_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cur_y += 40
    cv2.line(canvas, (text_x, cur_y-15), (panel_w-20, cur_y-15), (80,80,80), 1)

    # 系统状态
    cv2.putText(canvas, "SYSTEM STATUS", (text_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cur_y += 30
    cam_status = "MOVING" if camera_moving else "STATIC"
    cam_color = (0,0,255) if camera_moving else (0,255,0)
    cv2.putText(canvas, f"Camera: {cam_status}", (text_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cam_color, 1)
    cur_y += 25
    cv2.putText(canvas, f"Motion: {motion_state}", (text_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cur_y += 25
    cv2.putText(canvas, f"dx: {dx:.1f}px  dy: {dy:.1f}px", (text_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cur_y += 25
    cv2.putText(canvas, f"angle: {angle_deg:.1f}deg", (text_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cur_y += 25
    cv2.putText(canvas, f"residual: {residual:.1f}px", (text_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cur_y += 40
    cv2.line(canvas, (text_x, cur_y-15), (panel_w-20, cur_y-15), (80,80,80), 1)

    # 目标计数
    cv2.putText(canvas, "OBJECT COUNT", (text_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cur_y += 30
    for name, num in count.items():
        cv2.putText(canvas, f"{name}: {num}", (text_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
        cur_y += 25
    cur_y += 15
    cv2.line(canvas, (text_x, cur_y-15), (panel_w-20, cur_y-15), (80,80,80), 1)

    # 图例
    cv2.putText(canvas, "LEGEND", (text_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cur_y += 30
    legends = [
        ((0,255,0), "Static target"),
        ((0,0,255), "Moving target"),
        ((0,255,255), "New target"),
        ((128,128,128), "Occluded"),
        ((0,165,255), "Running"),
        ((0,0,255), "FALL RISK")
    ]
    for color, text in legends:
        cv2.rectangle(canvas, (text_x, cur_y-12), (text_x+20, cur_y+2), color, -1)
        cv2.putText(canvas, text, (text_x+30, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cur_y += 22
    cur_y += 10
    cv2.line(canvas, (text_x, cur_y-15), (panel_w-20, cur_y-15), (80,80,80), 1)

    # 底部提示
    cv2.putText(canvas, "Press 'q' to exit", (text_x, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128,128,128), 1)

    return canvas