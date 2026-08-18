"""FR05 逆行检测模块 — 可接入总线。"""
from .pipeline import WrongWayPipeline
from .wrong_way import WrongWayDetector

__all__ = ["WrongWayPipeline", "WrongWayDetector"]
__version__ = "0.1.0"
