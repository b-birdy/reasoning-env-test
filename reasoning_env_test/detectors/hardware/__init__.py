from .base import BaseHardwareDetector
from .cpu import CPUDetector
from .nvidia import NvidiaDetector
from .amd import AmdDetector
from .ascend import AscendDetector
from .kunlunxin import KunlunxinDetector
from .hygon import HygonDetector

__all__ = [
    "BaseHardwareDetector",
    "CPUDetector",
    "NvidiaDetector",
    "AmdDetector",
    "AscendDetector",
    "KunlunxinDetector",
    "HygonDetector",
    "detect_all",
]


def detect_all() -> list:
    """运行所有硬件检测器，返回检测结果列表。

    Returns:
        包含每个检测器结果的字典列表，遵循统一输出格式。
    """
    detectors = [
        CPUDetector(),
        NvidiaDetector(),
        AmdDetector(),
        AscendDetector(),
        KunlunxinDetector(),
        HygonDetector(),
    ]
    return [detector.detect() for detector in detectors]
