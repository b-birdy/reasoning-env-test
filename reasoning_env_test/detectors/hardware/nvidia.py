import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import BaseHardwareDetector


class NvidiaDetector(BaseHardwareDetector):
    """NVIDIA GPU 检测器。

    优先使用 pynvml 库获取 GPU 信息，不可用时回退到 nvidia-smi 命令。
    """

    def detect(self) -> Dict[str, Any]:
        result = self._empty_result("nvidia")

        gpus = self._detect_pynvml()
        if gpus is None:
            gpus = self._detect_nvidia_smi()

        if not gpus:
            return result

        result["model"] = gpus[0].get("name", "")
        result["memory_total_gb"] = gpus[0].get("memory_total_gb", 0.0)
        result["compute_units"] = len(gpus)
        result["details"] = {
            "gpu_count": len(gpus),
            "gpus": gpus,
        }
        return result

    def _detect_pynvml(self) -> Optional[List[Dict[str, Any]]]:
        """通过 pynvml 库检测 NVIDIA GPU。"""
        try:
            import pynvml  # type: ignore
        except ImportError:
            return None

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                gpus.append({
                    "index": i,
                    "name": name if isinstance(name, str) else name.decode("utf-8", errors="replace"),
                    "memory_total_gb": round(mem_info.total / (1024 ** 3), 2),
                    "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
                })
            pynvml.nvmlShutdown()
            return gpus
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return None

    def _detect_nvidia_smi(self) -> Optional[List[Dict[str, Any]]]:
        """通过 nvidia-smi 命令检测 NVIDIA GPU。"""
        if not shutil.which("nvidia-smi"):
            return None

        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,compute_cap",
                    "--format=csv,noheader",
                ],
                text=True, timeout=15, stderr=subprocess.DEVNULL,
            )
            gpus = []
            for line in output.strip().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue

                index = int(parts[0])
                name = parts[1]
                mem_str = parts[2]

                # 解析显存大小，如 "16384 MiB" 或 "80 GiB"
                mem_mib = 0.0
                if "MiB" in mem_str:
                    mem_mib = float(mem_str.replace("MiB", "").strip())
                elif "GiB" in mem_str:
                    mem_mib = float(mem_str.replace("GiB", "").strip()) * 1024

                memory_total_gb = round(mem_mib / 1024, 2)
                compute_cap = parts[3] if len(parts) >= 4 else ""

                gpus.append({
                    "index": index,
                    "name": name,
                    "memory_total_gb": memory_total_gb,
                    "compute_capability": compute_cap,
                })

            return gpus if gpus else None
        except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError):
            return None
