import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import BaseHardwareDetector


class AmdDetector(BaseHardwareDetector):
    """AMD GPU 检测器。

    通过 rocm-smi 命令获取 AMD GPU 信息。
    在非 Linux 平台或未安装 ROCm 时返回空结果。
    """

    def detect(self) -> Dict[str, Any]:
        result = self._empty_result("amd")

        if not shutil.which("rocm-smi"):
            return result

        gpus = self._detect_rocm_smi()
        if not gpus:
            return result

        result["model"] = gpus[0].get("name", "")
        result["memory_total_gb"] = gpus[0].get("memory_total_gb", 0.0)
        result["compute_units"] = gpus[0].get("compute_units", 0)
        result["details"] = {
            "gpu_count": len(gpus),
            "gpus": gpus,
        }
        return result

    def _detect_rocm_smi(self) -> Optional[List[Dict[str, Any]]]:
        """通过 rocm-smi 获取 GPU 信息。"""
        try:
            output = subprocess.check_output(
                ["rocm-smi", "--showallinfo"],
                text=True, timeout=30, stderr=subprocess.DEVNULL,
            )
            return self._parse_rocm_smi_output(output)
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    def _parse_rocm_smi_output(self, output: str) -> Optional[List[Dict[str, Any]]]:
        """解析 rocm-smi 输出。"""
        gpus = []
        current_gpu = None

        for line in output.splitlines():
            stripped = line.strip()

            # 检测 GPU 区块头：如 "GPU 0" 或 "GPU[0]"
            gpu_match = re.search(r'GPU\s+\[?(\d+)\]?', stripped)
            if gpu_match and ("===" in stripped or stripped.startswith("GPU")):
                if current_gpu is not None and "index" in current_gpu:
                    gpus.append(current_gpu)
                current_gpu = {
                    "index": int(gpu_match.group(1)),
                    "name": "",
                    "memory_total_gb": 0.0,
                    "compute_units": 0,
                }
                continue

            if current_gpu is None:
                continue

            if ":" not in stripped:
                continue

            key, value = stripped.split(":", 1)
            key_lower = key.strip().lower()
            value = value.strip()

            # 卡型号/设备名称
            if any(k in key_lower for k in ["card model", "device name", "device type", "name", "product"]):
                if value and not current_gpu["name"]:
                    current_gpu["name"] = value

            # 显存
            if any(k in key_lower for k in ["vram", "memory size", "memory"]):
                m = re.search(r'(\d+\.?\d*)\s*(MB|GB)', value, re.IGNORECASE)
                if m:
                    val = float(m.group(1))
                    if m.group(2).upper() == "MB":
                        current_gpu["memory_total_gb"] = round(val / 1024, 2)
                    else:
                        current_gpu["memory_total_gb"] = round(val, 2)

            # 计算单元
            if re.search(r'compute\s*units?\b', key_lower):
                m = re.search(r'(\d+)', value)
                if m:
                    current_gpu["compute_units"] = int(m.group(1))

        if current_gpu is not None and "index" in current_gpu:
            gpus.append(current_gpu)

        return gpus if gpus else None
