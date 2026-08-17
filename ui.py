"""
统一 UI 显示
============
OpenCV 画图形 + PIL 画中文，参考 vru 模块 visualization.py 风格。

布局:
  [左侧信息栏 320px] [中间视频] [右侧 BEV 俯视图 480px] [底部状态栏]
"""
import math
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from system_config import *
from ttc.core import build_ego_trajectory
from ttc.models import EgoState as TTCEgoState

# COCO 17 关键点骨架连接
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
    (5, 11), (6, 12), (11, 12),            # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),  # 下肢
]

# 行为颜色 (BGR)
_BEHAVIOR_COLORS = {
    "falling": (0, 0, 255),
    "running": (0, 165, 255),
    "wrong_way": (255, 0, 255),
    "suspect_wrong_way": (255, 100, 255),
    "normal": (0, 255, 0),
}

_BEHAVIOR_CN = {
    "normal": "正常",
    "running": "奔跑",
    "falling": "跌倒",
    "wrong_way": "逆行",
    "suspect_wrong_way": "疑似逆行",
}


class UnifiedUI:
    """统一 UI 渲染器。"""

    def __init__(self, show_trajectories: bool = False):
        self.show_trajectories = bool(show_trajectories)
        self.font_small = ImageFont.truetype(FONT_PATH, UI_FONT_SIZE_SMALL)
        self.font_normal = ImageFont.truetype(FONT_PATH, UI_FONT_SIZE_NORMAL)
        self.font_large = ImageFont.truetype(FONT_PATH, UI_FONT_SIZE_LARGE)

        # BEV 视图参数
        self.bev_range_front = 30.0   # 显示前方 30m
        self.bev_range_back = 5.0     # 后方 5m
        self.bev_range_side = 15.0    # 左右各 15m
        self.bev_scale = 0.0          # 像素/米，初始化时计算

        self.window_name = UI_WINDOW_NAME
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def render(self, result) -> np.ndarray:
        """渲染一帧，返回合成图像。"""
        frame = result.frame.copy()
        h, w = frame.shape[:2]

        # ---- 在视频帧上画检测结果 ----
        self._draw_detections(frame, result)

        # ---- BEV 俯视图 ----
        bev_img = self._draw_bev(result)
        bev_h, bev_w = bev_img.shape[:2]

        # ---- 左侧信息栏 ----
        panel_w = UI_PANEL_WIDTH
        panel_h = h
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        # ---- 合成 ----
        center_w = w
        total_w = panel_w + center_w + bev_w
        total_h = max(h, bev_h, panel_h)
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        canvas[0:panel_h, 0:panel_w] = panel
        canvas[0:h, panel_w:panel_w + w] = frame
        canvas[0:bev_h, panel_w + w:panel_w + w + bev_w] = bev_img

        # ---- PIL 画中文 ----
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        self._draw_left_panel(draw, result, panel_w, panel_h)
        self._draw_detection_labels(draw, result, panel_w)
        self._draw_bottom_bar(draw, result, total_w, total_h)

        canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return canvas

    def show(self, result):
        """渲染并显示。"""
        canvas = self.render(result)
        cv2.imshow(self.window_name, canvas)
        return canvas

    def _draw_detections(self, frame, result):
        """在视频帧上画检测框、标签、骨架。"""
        for obj in result.tracks:
            x1, y1, x2, y2 = obj["bbox"]
            tid = obj["track_id"]
            cls = obj["cls"]
            behavior = obj.get("behavior", "normal")
            color = _BEHAVIOR_COLORS.get(behavior, (0, 255, 0))

            # 丢失目标用虚线框（简化为半透明）
            is_lost = obj.get("track_state", 1) == 2
            thickness = 1 if is_lost else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 骨架
            kps = obj.get("keypoints")
            if kps is not None and cls == 0:
                self._draw_skeleton(frame, kps)

            # 朝向（逆行模块）
            face = obj.get("orient_face")
            if face in ("front", "rear"):
                cx = (x1 + x2) // 2
                arrow_len = 20
                if face == "front":
                    cv2.arrowedLine(frame, (cx, y1), (cx, y1 - arrow_len),
                                    (0, 200, 255), 2, tipLength=0.3)
                else:
                    cv2.arrowedLine(frame, (cx, y2), (cx, y2 + arrow_len),
                                    (200, 200, 0), 2, tipLength=0.3)

    def _draw_detection_labels(self, draw: ImageDraw.ImageDraw, result, x_offset: int):
        """用 PIL 在检测框旁画中文标签。"""
        for obj in result.tracks:
            x1, y1, x2, y2 = obj["bbox"]
            cls = obj["cls"]
            tid = obj["track_id"]
            behavior = obj.get("behavior", "normal")
            cls_name = CLASS_NAMES_CN.get(cls, str(cls))
            beh_name = _BEHAVIOR_CN.get(behavior, behavior)

            # 标签文字
            label = f"{cls_name}#{tid}"
            if behavior != "normal":
                label += f" {beh_name}"

            # 画在框上方
            tx = x_offset + x1
            ty = max(0, y1 - 22)

            # 文字背景
            bbox = draw.textbbox((tx, ty), label, font=self.font_small)
            draw.rectangle(bbox, fill=(0, 0, 0))
            color = _BEHAVIOR_COLORS.get(behavior, (0, 255, 0))
            # PIL 用 RGB，OpenCV 用 BGR，需要反转
            color_rgb = (color[2], color[1], color[0])
            draw.text((tx, ty), label, fill=color_rgb, font=self.font_small)

    def _draw_skeleton(self, frame, keypoints):
        """画 COCO 17 关键点骨架。"""
        for i, (a, b) in enumerate(_SKELETON):
            if a < len(keypoints) and b < len(keypoints):
                pa = keypoints[a]
                pb = keypoints[b]
                if pa[2] > 0.3 and pb[2] > 0.3:
                    color = (255, 200, 0) if i < 4 else (0, 200, 255) if i < 10 else (200, 0, 200)
                    cv2.line(frame, (int(pa[0]), int(pa[1])),
                             (int(pb[0]), int(pb[1])), color, 2)
        for kp in keypoints:
            if kp[2] > 0.3:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)

    def _draw_bev(self, result) -> np.ndarray:
        """画 BEV 俯视图。"""
        bev_w = UI_BEV_WIDTH
        bev_h = int(bev_w * (self.bev_range_front + self.bev_range_back) /
                    (2 * self.bev_range_side))
        img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
        img[:] = (25, 25, 35)

        # 比例
        self.bev_scale = bev_w / (2 * self.bev_range_side)
        cx = bev_w // 2
        cy = int(self.bev_range_front / (self.bev_range_front + self.bev_range_back) * bev_h)

        def world_to_pixel(x_fwd, y_left):
            px = int(cx + y_left * self.bev_scale)
            py = int(cy - x_fwd * self.bev_scale)
            return px, py

        # 网格
        for d in range(0, int(self.bev_range_front) + 1, 5):
            px1, py = world_to_pixel(d, -self.bev_range_side)
            px2, _ = world_to_pixel(d, self.bev_range_side)
            cv2.line(img, (px1, py), (px2, py), (50, 50, 60), 1)
        for d in range(-int(self.bev_range_side), int(self.bev_range_side) + 1, 5):
            px, py1 = world_to_pixel(-self.bev_range_back, d)
            _, py2 = world_to_pixel(self.bev_range_front, d)
            cv2.line(img, (px, py1), (px, py2), (50, 50, 60), 1)

        # 自车（三角形）
        ego_pts = np.array([
            world_to_pixel(2.4, 0),
            world_to_pixel(-2.4, -1.0),
            world_to_pixel(-2.4, 1.0),
        ], dtype=np.int32)
        cv2.fillPoly(img, [ego_pts], (0, 200, 0))

        if self.show_trajectories:
            self._draw_future_trajectories(img, result, world_to_pixel)

        # 目标
        for rr in result.risk_results:
            x, y = rr.position
            px, py = world_to_pixel(x, y)
            if 0 <= px < bev_w and 0 <= py < bev_h:
                color = RISK_COLORS.get(rr.risk_level, (128, 128, 128))
                radius = 6 if rr.risk_level in ("warning", "danger") else 4
                cv2.circle(img, (px, py), radius, color, -1)
                cv2.circle(img, (px, py), radius + 2, color, 1)

        return img

    def _draw_future_trajectories(self, img, result, world_to_pixel):
        """绘制现有TTC函数生成的自车CTRV和QCNet多模态目标轨迹。"""
        ego_state = TTCEgoState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            speed_mps=float(result.ego_speed_mps),
            yaw_rate_rps=float(result.ego_yaw_rate),
        )
        ego_traj = build_ego_trajectory(
            ego_state,
            horizon_s=TTC_HORIZON_S,
            dt_s=TTC_DT_S,
        )
        ego_pixels = [world_to_pixel(point.x, point.y) for point in ego_traj]
        if len(ego_pixels) >= 2:
            cv2.polylines(
                img,
                [np.asarray(ego_pixels, dtype=np.int32)],
                False,
                (255, 160, 40),
                2,
                cv2.LINE_AA,
            )

        predictions = getattr(result, "qcnet_predictions", {}) or {}
        palette = [
            (255, 0, 255),
            (0, 220, 255),
            (255, 120, 0),
            (180, 80, 255),
            (0, 180, 120),
            (120, 220, 255),
        ]
        for target_index, (track_id, prediction) in enumerate(predictions.items()):
            modes = list(getattr(prediction, "modes", ()) or ())
            if not modes:
                continue
            best_index = max(range(len(modes)), key=lambda index: modes[index].probability)
            color = palette[target_index % len(palette)]
            for mode_index, mode in enumerate(modes):
                pixels = [world_to_pixel(point.x, point.y) for point in mode.points]
                if len(pixels) < 2:
                    continue
                is_best = mode_index == best_index
                line_color = color if is_best else tuple(int(channel * 0.42) for channel in color)
                cv2.polylines(
                    img,
                    [np.asarray(pixels, dtype=np.int32)],
                    False,
                    line_color,
                    3 if is_best else 1,
                    cv2.LINE_AA,
                )

            best_mode = modes[best_index]
            endpoint = best_mode.points[-1]
            end_pixel = world_to_pixel(endpoint.x, endpoint.y)
            cv2.circle(img, end_pixel, 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                img,
                f"#{track_id} {best_mode.probability:.2f}",
                (end_pixel[0] + 5, end_pixel[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_left_panel(self, draw: ImageDraw.ImageDraw, result, w, h):
        """画左侧信息栏（PIL 中文）。"""
        x = 12
        y = 10
        line_h = 28

        # 标题
        draw.text((x, y), "系统状态", fill=(0, 255, 200), font=self.font_large)
        y += 45

        # FPS / 延迟
        draw.text((x, y), f"FPS: {result.fps:.1f}", fill=(255, 255, 255), font=self.font_normal)
        y += line_h
        draw.text((x, y), f"延迟: {result.process_ms:.0f} ms", fill=(255, 255, 255), font=self.font_normal)
        y += line_h
        draw.text((x, y), f"帧号: {result.frame_id}", fill=(200, 200, 200), font=self.font_small)
        y += line_h + 5

        # 传感器
        sensor_cn = {"normal": "正常", "partial": "部分异常", "degraded": "降级",
                     "unavailable": "不可用"}
        sc = sensor_cn.get(result.sensor_status, result.sensor_status)
        sc_color = (0, 255, 0) if result.sensor_status == "normal" else (0, 165, 255)
        draw.text((x, y), f"传感器: {sc}", fill=sc_color, font=self.font_normal)
        y += line_h
        draw.text((x, y), f"质量分: {result.sensor_score:.2f}", fill=(200, 200, 200), font=self.font_small)
        y += line_h + 5

        # 自车
        draw.text((x, y), f"自车速度: {self.ego_speed_kmh(result):.0f} km/h",
                  fill=(255, 255, 255), font=self.font_normal)
        y += line_h + 5

        # QCNet
        if result.qcnet_active:
            model_count = len(result.qcnet_predictions)
            fallback_count = sum(
                1 for rr in result.risk_results
                if rr.prediction_source == "constant_velocity"
            )
            label = f"QCNet运行: {model_count}目标"
            if fallback_count:
                label += f" / 回退{fallback_count}"
            draw.text((x, y), label, fill=(0, 255, 0), font=self.font_small)
        elif result.qcnet_ready and result.qcnet_error:
            draw.text((x, y), "轨迹预测: 匀速降级", fill=(255, 180, 0), font=self.font_small)
        elif result.qcnet_ready:
            draw.text((x, y), "轨迹预测: 等待有效目标", fill=(0, 200, 255), font=self.font_small)
        else:
            draw.text((x, y), f"预热中 {result.qcnet_history_frames}/{QCNET_MIN_HISTORY}",
                      fill=(0, 200, 255), font=self.font_normal)
        y += line_h + 10

        # 分割线
        draw.line([(x, y), (w - 12, y)], fill=(80, 80, 80), width=1)
        y += 10

        # 风险目标列表
        draw.text((x, y), "风险目标", fill=(0, 255, 200), font=self.font_normal)
        y += 30

        risk_targets = [r for r in result.risk_results if r.risk_level != "safe"]
        if not risk_targets:
            draw.text((x, y), "无风险目标", fill=(100, 200, 100), font=self.font_small)
        else:
            for rr in risk_targets[:8]:
                level_cn = RISK_NAMES_CN.get(rr.risk_level, rr.risk_level)
                color = tuple(reversed(RISK_COLORS.get(rr.risk_level, (128, 128, 128))))
                ttc_str = f"{rr.ttc_ms}ms" if rr.ttc_ms is not None else "∞"
                draw.text((x, y), f"#{rr.track_id} {level_cn} TTC:{ttc_str}",
                          fill=color, font=self.font_small)
                y += 24
                if y > h - 80:
                    break

    def _draw_bottom_bar(self, draw: ImageDraw.ImageDraw, result, w, h):
        """画底部状态栏。"""
        bar_h = 35
        y = h - bar_h
        draw.rectangle([(0, y), (w, h)], fill=(40, 40, 45))

        # 最高风险等级
        max_risk = "safe"
        for rr in result.risk_results:
            if RISK_LEVELS_ORDER.index(rr.risk_level) > RISK_LEVELS_ORDER.index(max_risk):
                max_risk = rr.risk_level
        max_cn = RISK_NAMES_CN.get(max_risk, max_risk)
        color = tuple(reversed(RISK_COLORS.get(max_risk, (128, 128, 128))))

        n_targets = len(result.tracks)
        text = f"  目标数: {n_targets}    最高风险: {max_cn}    按 Q 退出"
        draw.text((10, y + 5), text, fill=(255, 255, 255), font=self.font_normal)

    @staticmethod
    def ego_speed_kmh(result):
        """从 pipeline 配置获取自车速度（km/h）。"""
        # result 中不直接存 ego_speed，这里返回 0，由 pipeline 注入
        return getattr(result, "ego_speed_kmh", 0.0)


# 风险等级排序（用于找最高风险）
RISK_LEVELS_ORDER = ["safe", "notice", "warning", "danger"]
