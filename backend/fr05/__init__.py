"""FR-05：特定异常行为识别。"""

from .interfaces import FR05Event, FR05Result

__all__ = ["FR05Event", "FR05Recognizer", "FR05Result"]


def __getattr__(name):
    """延迟加载模型相关依赖，接口定义可在无推理环境中独立使用。"""
    if name == "FR05Recognizer":
        from .recognizer import FR05Recognizer

        return FR05Recognizer
    raise AttributeError(name)
