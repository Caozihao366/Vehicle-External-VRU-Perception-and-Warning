"""统一推理设备选择。"""
from __future__ import annotations

import torch


def resolve_device_str(device: str = "auto") -> str:
    d = str(device).lower().strip()
    if d in ("auto", "gpu"):
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if d in ("mps", "metal"):
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if d.isdigit() or d.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda:0" if d.isdigit() else d
        return "cpu"
    return d or "cpu"


def resolve_torch_device(device: str = "auto") -> torch.device:
    return torch.device(resolve_device_str(device))
