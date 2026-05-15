import shutil
from typing import Any, Dict

from .base import BaseHardwareDetector


class KunlunxinDetector(BaseHardwareDetector):
    """昆仑芯 XPU 检测器（预留接口）。

    检测 xpu-smi 管理工具是否存在，返回占位信息。
    """

    def detect(self) -> Dict[str, Any]:
        result = self._empty_result("kunlunxin")

        if shutil.which("xpu-smi"):
            result["model"] = "Kunlunxin XPU (placeholder)"
            result["details"] = {
                "command_found": "xpu-smi",
                "message": "Kunlunxin XPU 检测到（预留占位）",
            }

        return result
