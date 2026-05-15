import shutil
from typing import Any, Dict

from .base import BaseHardwareDetector


class HygonDetector(BaseHardwareDetector):
    """海光 DCU 检测器（预留接口）。

    检测 hygon-smi 或 dcu-smi 管理工具是否存在，返回占位信息。
    """

    def detect(self) -> Dict[str, Any]:
        result = self._empty_result("hygon")

        hygon_commands = ["hygon-smi", "dcu-smi"]
        found_commands = [cmd for cmd in hygon_commands if shutil.which(cmd)]

        if found_commands:
            result["model"] = "Hygon DCU (placeholder)"
            result["details"] = {
                "commands_found": found_commands,
                "message": "海光 DCU 检测到（预留占位）",
            }

        return result
