"""YOLOPv2 车道线 / 可行驶区域推理封装。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .device_utils import resolve_torch_device

ROOT = Path(__file__).resolve().parent
YOLOPV2_DIR = ROOT / "third_party" / "YOLOPv2"
if str(YOLOPV2_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOPV2_DIR))

from utils.utils import driving_area_mask, lane_line_mask, letterbox  # noqa: E402

# 与官方 demo LoadImages 一致：先统一到 1280x720，再 letterbox
INFER_W, INFER_H = 1280, 720


def _resolve_device(device: str) -> torch.device:
    return resolve_torch_device(device)


class YOLOPv2Lane:
    def __init__(self, weights: str, device: str = "cpu", img_size: int = 640):
        self.weights = str(weights)
        self.img_size = img_size
        self.device = _resolve_device(device)
        self.half = self.device.type == "cuda"
        if not Path(self.weights).is_file():
            raise FileNotFoundError(
                f"未找到 YOLOPv2 权重: {self.weights}\n"
                "请放到 YOLOPv2/data/weights/yolopv2.pt\n"
                "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
            )

        self.model = torch.jit.load(self.weights, map_location=self.device)
        self.model = self.model.to(self.device)
        if self.half:
            self.model.half()
        self.model.eval()

        dummy = torch.zeros(1, 3, img_size, img_size, device=self.device)
        dummy = dummy.half() if self.half else dummy.float()
        with torch.no_grad():
            # warmup：官方用方图；实际输入可能是 letterbox 非方图
            try:
                self.model(dummy)
            except Exception:
                pass

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray):
        """返回与原图同尺寸的 (lane_mask, da_mask)，uint8，前景为 1。"""
        h0, w0 = frame_bgr.shape[:2]
        img0 = cv2.resize(frame_bgr, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, self.img_size, stride=32, auto=True)[0]
        img_in = img[:, :, ::-1].transpose(2, 0, 1)
        img_in = np.ascontiguousarray(img_in)
        tensor = torch.from_numpy(img_in).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        _det, seg, ll = self.model(tensor)
        da_mask = driving_area_mask(seg)  # 官方输出对齐 720x1280
        ll_mask = lane_line_mask(ll)

        da_mask = self._to_uint_mask(da_mask, h0, w0)
        ll_mask = self._to_uint_mask(ll_mask, h0, w0)
        return ll_mask, da_mask

    @staticmethod
    def _to_uint_mask(mask: np.ndarray, h0: int, w0: int) -> np.ndarray:
        if mask is None or mask.size == 0:
            return np.zeros((h0, w0), dtype=np.uint8)
        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.uint8)
        if mask.shape[0] != INFER_H or mask.shape[1] != INFER_W:
            mask = cv2.resize(mask, (INFER_W, INFER_H), interpolation=cv2.INTER_NEAREST)
        if (h0, w0) != (INFER_H, INFER_W):
            mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
        return mask


def _fit_heading_up(xs: np.ndarray, ys: np.ndarray):
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(pts) < 20:
        return None
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    dx, dy = float(line[0, 0]), float(line[1, 0])
    if dy > 0:
        dx, dy = -dx, -dy
    heading = float(np.degrees(np.arctan2(-dy, dx)))
    # 前视场景只接受接近「朝上」的结果（60°~120°）
    if 60.0 <= heading <= 120.0:
        return heading
    return None


def estimate_lane_heading_deg(
    lane_mask: np.ndarray,
    point_xy=None,
    search_half_w: int = 160,
    min_pixels: int = 40,
):
    """从车道线 mask 估计局部道路朝向（度）。前视约定：前进=朝图像上方(~90°)。"""
    if lane_mask is None or lane_mask.size == 0:
        return None

    h, w = lane_mask.shape[:2]
    y1, y2 = int(h * 0.50), h

    headings = []
    # 分左右拟合，避免整幅 V 形车道被拟成水平
    bands = [(0.10, 0.45), (0.55, 0.90)]
    if point_xy is not None:
        x0 = int(point_xy[0])
        bands.insert(0, (
            max(0.0, (x0 - search_half_w) / w),
            min(1.0, (x0 + search_half_w) / w),
        ))

    for a, b in bands:
        x1, x2 = int(w * a), int(w * b)
        if x2 - x1 < 20:
            continue
        roi = lane_mask[y1:y2, x1:x2]
        yy, xx = np.where(roi > 0)
        if len(xx) < min_pixels:
            continue
        heading = _fit_heading_up(xx.astype(np.float32) + x1, yy.astype(np.float32) + y1)
        if heading is not None:
            headings.append(heading)

    if not headings:
        return 90.0  # 前视默认：道路朝上
    return float(np.median(headings))


def build_ego_lane_mask(lane_mask: np.ndarray | None, da_mask: np.ndarray | None = None):
    """用近处车道线拟合左右边界，沿方向外推填充本车道。

    思路：
      1) 只在画面底部截取左右车道线点（离自车近、更稳）
      2) 各拟一条直线，从底边沿方向延伸到画面顶端
      3) 若左右线提前相交（灭点），交点以上不再延伸、不再填充
      4) 不做可行驶区域 flood-fill（da_mask 保留参数兼容，已不使用）

    近处线不足时返回 None（宁可不报，也不涂整片路）。
    """
    build_ego_lane_mask.last_source = "none"
    # da_mask 不再参与本车道估计
    _ = da_mask

    if lane_mask is None or lane_mask.size == 0 or lane_mask.sum() < 80:
        return None

    h, w = lane_mask.shape[:2]
    cx = w // 2

    # 只采底部带：默认下半部偏下（更稳）
    y_fit_top = int(h * 0.58)
    y_fit_bot = h - 1
    min_pts = 8
    min_width = 12
    max_width = int(w * 0.75)

    ll = cv2.dilate(
        (lane_mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)),
        1,
    )

    left_pts, right_pts = [], []
    for y in range(y_fit_bot, y_fit_top - 1, -1):
        xs = np.where(ll[y] > 0)[0]
        if xs.size == 0:
            continue
        left = xs[xs <= cx]
        right = xs[xs >= cx]
        if left.size == 0 or right.size == 0:
            continue
        # 取离中心最近的左/右线，避免抓到更外侧护栏/邻道
        lx, rx = int(left.max()), int(right.min())
        width = rx - lx
        if min_width <= width <= max_width:
            left_pts.append((lx, y))
            right_pts.append((rx, y))

    if len(left_pts) < min_pts or len(right_pts) < min_pts:
        return None

    la = np.array(left_pts, np.float32)
    ra = np.array(right_pts, np.float32)
    # x = a*y + b
    lf = np.poly1d(np.polyfit(la[:, 1], la[:, 0], 1))
    rf = np.poly1d(np.polyfit(ra[:, 1], ra[:, 0], 1))

    # 底部宽度合理性检查
    lx_b = float(lf(y_fit_bot))
    rx_b = float(rf(y_fit_bot))
    if not (lx_b < cx < rx_b and (rx_b - lx_b) >= min_width):
        return None

    # 灭点：lf(y)=rf(y) → (aL-aR)*y + (bL-bR)=0
    a_l, b_l = float(lf.c[0]), float(lf.c[1])
    a_r, b_r = float(rf.c[0]), float(rf.c[1])
    y_vanish = None
    denom = a_l - a_r
    if abs(denom) > 1e-6:
        yv = -(b_l - b_r) / denom
        # 只接受在底边之上、画面内的交点（透视汇合）
        if 0.0 <= yv < float(y_fit_bot):
            y_vanish = int(yv)

    # 从底往上填：碰到交点或宽度过窄则停
    y_stop = 0 if y_vanish is None else max(0, y_vanish)
    ego = np.zeros((h, w), np.uint8)
    for y in range(y_fit_bot, y_stop - 1, -1):
        lx = int(round(float(lf(y))))
        rx = int(round(float(rf(y))))
        if rx < lx:
            lx, rx = rx, lx
        # 提前汇合：宽度不够就停，不再往天空延伸
        if (rx - lx) < min_width:
            break
        lx = int(np.clip(lx, 0, w - 1))
        rx = int(np.clip(rx, 0, w - 1))
        # 外推后仍应夹住画面中心（本车道）
        if not (lx < cx < rx):
            break
        ego[y, lx : rx + 1] = 1

    if ego.sum() < 200:
        return None

    ego = cv2.dilate(
        ego, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7)), 1
    )
    build_ego_lane_mask.last_source = "bottom_extrapolate"
    if y_vanish is not None:
        build_ego_lane_mask.last_source = "bottom_extrapolate+vanish"
    return ego


def ego_lane_relation(ego_mask, bbox, bottom_center=None, invade_overlap=0.08):
    """in_ego / invade_ego / outside / unknown"""
    if ego_mask is None or ego_mask.size == 0:
        return "unknown"
    h, w = ego_mask.shape[:2]
    ego = cv2.dilate(
        (ego_mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 15)),
        1,
    )
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return "outside"
    if bottom_center is None:
        bx, by = (x1 + x2) // 2, y2
    else:
        bx, by = int(bottom_center[0]), int(bottom_center[1])
    bx = min(max(bx, 0), w - 1)
    by = min(max(by, 0), h - 1)
    if ego[by, bx] > 0:
        return "in_ego"
    band_y1 = max(y1, y2 - max(8, int((y2 - y1) * 0.45)))
    band = ego[band_y1:y2, x1:x2]
    if band.size and float(band.mean()) >= invade_overlap:
        return "invade_ego"
    return "outside"


def overlay_lane(
    frame_bgr: np.ndarray,
    lane_mask: np.ndarray,
    da_mask: np.ndarray,
    ego_mask: np.ndarray | None = None,
    alpha: float = 0.35,
):
    out = frame_bgr.copy()
    color = np.zeros_like(out)
    if da_mask is not None:
        color[da_mask > 0] = (0, 180, 0)
    if ego_mask is not None:
        color[ego_mask > 0] = (180, 180, 0)  # 青色 = 本车道
    if lane_mask is not None:
        color[lane_mask > 0] = (0, 0, 255)
    mask = color.any(axis=2)
    out[mask] = (out[mask] * (1 - alpha) + color[mask] * alpha).astype(np.uint8)
    return out
