import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import BaseHardwareDetector


class AscendDetector(BaseHardwareDetector):
    """昇腾 NPU 检测器。

    通过 npu-smi 命令获取昇腾 AI 处理器信息。
    在命令不存在时返回空结果。
    """

    def detect(self) -> Dict[str, Any]:
        result = self._empty_result("ascend")

        if not shutil.which("npu-smi"):
            return result

        npus = self._detect_npu_smi()
        if not npus:
            return result

        result["model"] = npus[0].get("name", "")
        result["memory_total_gb"] = npus[0].get("memory_total_gb", 0.0)
        result["compute_units"] = npus[0].get("compute_units", 0)
        result["details"] = {
            "npu_count": len(npus),
            "npus": npus,
        }
        return result

    def _detect_npu_smi(self) -> Optional[List[Dict[str, Any]]]:
        """通过 npu-smi 获取 NPU 信息。"""
        try:
            output = subprocess.check_output(
                ["npu-smi", "info", "-t", "board"],
                text=True, timeout=15, stderr=subprocess.DEVNULL,
            )
            return self._parse_npu_smi_output(output)
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    def _parse_npu_smi_output(self, output: str) -> Optional[List[Dict[str, Any]]]:
        """解析 npu-smi 输出。"""
        npus = []
        current_npu = None

        for line in output.splitlines():
            stripped = line.strip()

            # 检测 NPU ID: "NPU ID: 0" 或 "Device ID: 0"
            id_match = re.search(r'(?:NPU\s*ID|Device\s*ID)\s*:\s*(\d+)', stripped, re.IGNORECASE)
            if id_match:
                if current_npu is not None and "index" in current_npu:
                    npus.append(current_npu)
                current_npu = {
                    "index": int(id_match.group(1)),
                    "name": "",
                    "memory_total_gb": 0.0,
                    "compute_units": 0,
                }
                continue

            if current_npu is None:
                continue

            # 产品名称/型号
            name_match = re.search(r'(?:Name|Model|Product\s*Name)\s*:\s*(.+)', stripped, re.IGNORECASE)
            if name_match:
                current_npu["name"] = name_match.group(1).strip()
                continue

            # 内存信息
            mem_match = re.search(r'(\d+\.?\d*)\s*(GB|MB)\s*(?:Total|Memory)', stripped, re.IGNORECASE)
            if mem_match:
                val = float(mem_match.group(1))
                if mem_match.group(2).upper() == "MB":
                    current_npu["memory_total_gb"] = round(val / 1024, 2)
                else:
                    current_npu["memory_total_gb"] = round(val, 2)
                continue

            # AI Core 数量（昇腾的计算单元）
            core_match = re.search(r'(?:AI\s*Core|Core\s*Count)\s*:\s*(\d+)', stripped, re.IGNORECASE)
            if core_match:
                current_npu["compute_units"] = int(core_match.group(1))

        if current_npu is not None and "index" in current_npu:
            npus.append(current_npu)

        return npus if npus else None
