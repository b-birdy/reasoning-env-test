import os
import platform
import subprocess
from typing import Any, Dict, Optional

from .base import BaseHardwareDetector


class CPUDetector(BaseHardwareDetector):
    """CPU 检测器。

    检测 CPU 核心数、架构、频率和系统总内存。
    优先使用 psutil，不可用时回退到平台标准库和系统命令。
    """

    def detect(self) -> Dict[str, Any]:
        result = self._empty_result("cpu")

        arch = platform.machine() or platform.architecture()[0] or ""
        model = self._get_cpu_model()
        cores = self._get_cpu_count()
        freq = self._get_cpu_freq()
        memory = self._get_total_memory()

        result["model"] = model
        result["memory_total_gb"] = round(memory, 2)
        result["compute_units"] = cores
        result["details"] = {
            "architecture": arch,
            "logical_cores": cores,
            "frequency_mhz": freq,
            "os": platform.system(),
            "os_version": platform.version(),
        }
        return result

    def _get_cpu_model(self) -> str:
        """获取 CPU 型号名称。"""
        # 尝试通过 platform.processor() 获取
        try:
            proc = platform.processor()
            if proc:
                return proc
        except Exception:
            pass

        # Windows: 通过 wmic 获取
        if platform.system() == "Windows":
            try:
                output = subprocess.check_output(
                    ["wmic", "cpu", "get", "name", "/format:value"],
                    text=True, timeout=5, stderr=subprocess.DEVNULL,
                )
                for line in output.splitlines():
                    if "=" in line:
                        return line.split("=", 1)[1].strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        # Linux: 通过 /proc/cpuinfo 获取
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except (FileNotFoundError, IOError):
                pass

        # macOS: 通过 sysctl 获取
        if platform.system() == "Darwin":
            try:
                output = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True, timeout=5, stderr=subprocess.DEVNULL,
                )
                return output.strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        return ""

    def _get_cpu_count(self) -> int:
        """获取逻辑 CPU 核心数量。"""
        try:
            import psutil  # type: ignore
            return psutil.cpu_count(logical=True) or os.cpu_count() or 0
        except ImportError:
            return os.cpu_count() or 0

    def _get_cpu_freq(self) -> float:
        """获取 CPU 当前频率（MHz）。"""
        try:
            import psutil  # type: ignore
            freq = psutil.cpu_freq()
            if freq:
                return round(freq.current, 2)
        except (ImportError, AttributeError):
            pass

        # Linux: 从 /proc/cpuinfo 获取
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("cpu MHz"):
                            return round(float(line.split(":", 1)[1].strip()), 2)
            except (FileNotFoundError, IOError, ValueError):
                pass

        return 0.0

    def _get_total_memory(self) -> float:
        """获取系统总内存（GB）。"""
        try:
            import psutil  # type: ignore
            mem = psutil.virtual_memory()
            return mem.total / (1024 ** 3)
        except ImportError:
            pass

        # Windows: 通过 wmic 获取
        if platform.system() == "Windows":
            try:
                output = subprocess.check_output(
                    ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/format:value"],
                    text=True, timeout=5, stderr=subprocess.DEVNULL,
                )
                for line in output.splitlines():
                    if "=" in line:
                        return int(line.split("=", 1)[1].strip()) / (1024 ** 3)
            except (subprocess.SubprocessError, FileNotFoundError, ValueError):
                pass

        # Linux: 从 /proc/meminfo 获取
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return kb / (1024 ** 2)  # KB -> GB
            except (FileNotFoundError, IOError, ValueError):
                pass

        return 0.0
