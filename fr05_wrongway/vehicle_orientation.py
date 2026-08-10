"""车辆朝向分类：front / rear（框内裁剪）。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# 默认二分类；实际以 checkpoint['classes'] 为准
ORIENT_CLASSES = ("front", "rear")


def build_orientation_model(num_classes: int = 2, pretrained_backbone: bool = True) -> nn.Module:
    """构建 ResNet18；优先本地缓存权重，避免 SSL 下载失败。"""
    model = models.resnet18(weights=None)
    if pretrained_backbone:
        local = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
        project_local = Path(__file__).resolve().parent / "weights" / "resnet18-f37072fd.pth"
        weight_path = local if local.is_file() else project_local
        if weight_path.is_file():
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            print(f"[orient] backbone from {weight_path}")
        else:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
                print("[orient] backbone from torchvision download")
            except Exception as e:
                print(f"[orient] 预训练下载失败，改用随机初始化: {e}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_infer_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


from .device_utils import resolve_torch_device


class VehicleOrientationClassifier:
    """对车辆检测框做 front/rear 分类。无权重时 enabled=False。"""

    def __init__(
        self,
        weights_path: str,
        device: str = "mps",
        img_size: int = 224,
        conf_thresh: float = 0.55,
    ):
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.transform = get_infer_transform(img_size)
        self.device = resolve_torch_device(device)
        self.enabled = False
        self.model = None
        self.classes = list(ORIENT_CLASSES)

        path = Path(weights_path)
        if not path.is_file():
            print(f"[orient] 未找到权重 {weights_path}，朝向主判据暂不可用（将用运动辅助）")
            print("[orient] 训练: ./.venv/bin/python train_orientation.py --data data/car_direction_cls")
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "classes" in ckpt:
            self.classes = list(ckpt["classes"])
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(ckpt, dict) and "img_size" in ckpt:
            self.img_size = int(ckpt["img_size"])
            self.transform = get_infer_transform(self.img_size)

        self.model = build_orientation_model(num_classes=len(self.classes), pretrained_backbone=False)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.enabled = True
        print(f"[orient] loaded {path} on {self.device} classes={self.classes}")

    @torch.no_grad()
    def predict_crop(self, crop_bgr: np.ndarray):
        if not self.enabled or self.model is None:
            return "unknown", 0.0, {}
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown", 0.0, {}
        if crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 16:
            return "unknown", 0.0, {}

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.classes[idx]
        prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        if conf < self.conf_thresh:
            return "unknown", conf, prob_dict
        return label, conf, prob_dict

    def predict_bbox(self, frame_bgr: np.ndarray, bbox, pad_ratio: float = 0.05):
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad_ratio), int(bh * pad_ratio)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        return self.predict_crop(frame_bgr[y1:y2, x1:x2])


def face_to_heading_deg(face: str, lane_heading_deg: float | None = None) -> float | None:
    """rear→与车道同向；front→与车道反向。"""
    if face not in ("front", "rear"):
        return None
    base = 90.0 if lane_heading_deg is None else float(lane_heading_deg)
    if face == "rear":
        return base
    return base + 180.0
