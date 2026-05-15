import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import BaseHardwareDetector


class KunlunxinDetector(BaseHardwareDetector):
    """昆仑芯 XPU 检测器。

    通过 xpu-smi 命令检测昆仑芯 P800 等 XPU 设备的数量和显存信息。
    支持两种输出格式：
      - key-value 格式（xpu-smi query -a）：每块卡一个信息块，Key: Value 形式
      - table 格式（xpu-smi query）：表格形式，包含 DevID、Name、Memory 等列
    """

    def detect(self) -> Dict[str, Any]:
        result = self._empty_result("kunlunxin")

        if not shutil.which("xpu-smi"):
            return result

        # 优先使用详细 key-value 格式
        xpus = self._parse_xpu_smi_detailed()
        if not xpus:
            # 回退到 table 格式
            xpus = self._parse_xpu_smi_table()

        if not xpus:
            return result

        result["model"] = xpus[0].get("name", "")
        result["memory_total_gb"] = xpus[0].get("memory_total_gb", 0.0)
        result["compute_units"] = len(xpus)
        result["details"] = {
            "xpu_count": len(xpus),
            "xpus": xpus,
        }
        return result

    # ------------------------------------------------------------------
    # Key-value 格式解析（xpu-smi query -a）
    # ------------------------------------------------------------------

    def _parse_xpu_smi_detailed(self) -> Optional[List[Dict[str, Any]]]:
        """调用 xpu-smi query -a 并解析 key-value 格式输出。"""
        try:
            output = subprocess.check_output(
                ["xpu-smi", "query", "-a"],
                text=True, timeout=15, stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        return self._parse_detailed_output(output)

    def _parse_detailed_output(self, output: str) -> Optional[List[Dict[str, Any]]]:
        """解析 key-value 格式的 xpu-smi 输出。

        典型输出：
            Device Index: 0
            Device Name: Kunlun XPU P800
            Memory Total: 16384 MB

            Device Index: 1
            Device Name: Kunlun XPU P800
            Memory Total: 16384 MB
        """
        xpus: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}

        for line in output.strip().splitlines():
            line = line.strip()
            if not line:
                # 空行作为设备块分隔符
                if current:
                    xpus.append(current)
                    current = {}
                continue

            match = re.match(r"^(.+?)\s*:\s*(.+)$", line)
            if not match:
                continue

            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            if not value:
                continue

            # 检测到新的设备索引 -> 开始新块
            if key in ("device index", "devid", "device id"):
                if current:
                    xpus.append(current)
                idx_match = re.search(r"\d+", value)
                current = {"index": int(idx_match.group()) if idx_match else 0}
            elif "device name" in key or "device model" in key:
                current["name"] = value
            elif "memory total" in key or key == "memory":
                current["memory_total_gb"] = self._parse_memory_to_gb(value)

        if current:
            xpus.append(current)

        return self._normalize_xpus(xpus)

    # ------------------------------------------------------------------
    # Table 格式解析（xpu-smi query）
    # ------------------------------------------------------------------

    def _parse_xpu_smi_table(self) -> Optional[List[Dict[str, Any]]]:
        """调用 xpu-smi query 并解析 table 格式输出。"""
        try:
            output = subprocess.check_output(
                ["xpu-smi", "query"],
                text=True, timeout=15, stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        return self._parse_table_output(output)

    def _parse_table_output(self, output: str) -> Optional[List[Dict[str, Any]]]:
        """解析 table 格式的 xpu-smi 输出。

        典型输出：
            +--------+------------------+--------------+
            | DevID  | Name             | Memory(MB)   |
            +--------+------------------+--------------+
            | 0      | Kunlun XPU P800  | 16384        |
            | 1      | Kunlun XPU P800  | 16384        |
            +--------+------------------+--------------+
        """
        lines = output.strip().splitlines()

        # 定位表头行（包含 DevID 和 Name）
        header_idx = None
        for i, line in enumerate(lines):
            if "|" in line and "DevID" in line and "Name" in line:
                header_idx = i
                break

        if header_idx is None:
            return None

        xpus: List[Dict[str, Any]] = []
        # 数据行从 header + 2 开始（跳过表头分隔线）
        for line in lines[header_idx + 2:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("+"):
                break

            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) < 2:
                continue

            index = int(parts[0]) if parts[0].isdigit() else 0
            name = parts[1]
            memory = self._parse_memory_to_gb(parts[2]) if len(parts) >= 3 else 0.0
            xpus.append({
                "index": index,
                "name": name,
                "memory_total_gb": memory,
            })

        return self._normalize_xpus(xpus)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_memory_to_gb(value: str) -> float:
        """将显存字符串转为 GB 浮点数。

        支持的输入格式：
            "16384 MB"      -> 16.0
            "16384 MiB"     -> 16.0
            "16 GB"         -> 16.0
            "16 GiB"        -> 16.0
            "16384"         -> 16.0  （无单位时按 MB 处理）
        """
        value = value.strip()
        match = re.search(r"([\d.]+)\s*(MB|GB|MiB|GiB)?", value, re.IGNORECASE)
        if match:
            num = float(match.group(1))
            unit = (match.group(2) or "MB").upper()
            if unit in ("GB", "GIB"):
                return round(num, 2)
            # MiB 和 MiB 都按 1024 换算
            return round(num / 1024, 2)
        return 0.0

    @staticmethod
    def _normalize_xpus(xpus: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """验证并补全 XPU 列表字段。"""
        if not xpus:
            return None

        for xpu in xpus:
            xpu.setdefault("index", 0)
            xpu.setdefault("name", "Kunlun XPU")
            xpu.setdefault("memory_total_gb", 0.0)

        return xpus
