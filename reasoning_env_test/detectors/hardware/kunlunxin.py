import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import BaseHardwareDetector


class KunlunxinDetector(BaseHardwareDetector):
    """昆仑芯 XPU 检测器。

    通过 xpu-smi 命令检测昆仑芯 P800 等 XPU 设备。
    真实服务器上实测格式：
      - xpu-smi -q ：详细 key-value 格式，每块卡输出完整属性
      - xpu-smi -m ：machine-readable 单行格式
      - xpu-smi（无参数）：表格摘要（含 Memory-Usage 列）
    """

    # xpu-smi machine-readable 字段位置
    _MR_MEM_TOTAL_POS = 18  # mem_total (MB)
    _MR_MEM_USED_POS = 17   # mem_used (MB)

    def detect(self) -> Dict[str, Any]:
        result = self._empty_result("kunlunxin")

        if not shutil.which("xpu-smi"):
            return result

        # 优先级: xpu-smi -q (详细) > xpu-smi -m (机器可读) > xpu-smi (摘要表格)
        xpus = self._parse_xpu_smi_q()
        if not xpus:
            xpus = self._parse_xpu_smi_m()
        if not xpus:
            xpus = self._parse_xpu_smi_summary()

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
    # xpu-smi -q : 详细 key-value 格式
    # ------------------------------------------------------------------

    def _parse_xpu_smi_q(self) -> Optional[List[Dict[str, Any]]]:
        """调用 xpu-smi -q 并解析详细 key-value 输出。

        典型输出：
            ==============XPUSMI LOG==============

            Timestamp                                 : Fri May 15 14:22:01 2026
            Driver Version                            : 5.0.21.21
            XPU-RT Version                            : 10.2

            Attached XPUs                             : 8
            XPU 00000000:29:00.0
                Product Name                          : P800 OAM
                ...
                Memory Usage
                    Total                             : 98304 MiB
                    Used                              : 84504 MiB
                    Free                              : 13800 MiB
        """
        try:
            output = subprocess.check_output(
                ["xpu-smi", "-q"],
                text=True, timeout=15, stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        return self._parse_q_output(output)

    def _parse_q_output(self, output: str) -> Optional[List[Dict[str, Any]]]:
        """解析 xpu-smi -q 的 key-value 输出。

        格式特点：
          - 每个 XPU 块以 'XPU <pci_addr>' 开头（必须是行首的 'XPU ' 后跟十六进制地址）
          - 各部分标题（如 Memory Usage、Firmware Version）独占一行，无冒号
          - key-value 格式：'Key : Value'
          - 缩进表示嵌套
        """
        xpus: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        in_memory_section = False

        for line in output.splitlines():
            stripped = line.strip()

            # ── 检测 XPU 块开头: 行首 "XPU " + PCI 地址格式 ──
            # 精确匹配 'XPU 00000000:29:00.0'，排除 'XPU Link Info' 等
            xpu_match = re.match(r"^XPU\s+[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]", stripped)
            if xpu_match:
                if current is not None:
                    xpus.append(current)
                current = {"index": len(xpus), "name": "Kunlun XPU", "memory_total_gb": 0.0}
                in_memory_section = False
                continue

            # ── 没有活跃 XPU 块时，只跟踪 Memory Usage 切换 ──
            if current is None:
                if stripped.rstrip(":") == "Memory Usage":
                    in_memory_section = True
                elif stripped and ":" not in stripped and not stripped.startswith("="):
                    in_memory_section = False
                continue

            # ── 活跃 XPU 块内 ──

            # 没有冒号的行 -> 新部分标题，重置 in_memory_section
            if ":" not in stripped and stripped and not stripped.startswith("="):
                in_memory_section = stripped.rstrip(":") == "Memory Usage"
                continue

            # 解析 key: value
            kv_match = re.match(r"^\s*(.+?)\s*:\s*(.+)$", line)
            if not kv_match:
                continue

            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip()

            if key == "Product Name":
                current["name"] = value
            elif in_memory_section and key == "Total":
                current["memory_total_gb"] = self._parse_memory_to_gb(value)

        if current is not None:
            xpus.append(current)

        return self._normalize_xpus(xpus)

    # ------------------------------------------------------------------
    # xpu-smi -m : machine-readable 单行格式
    # ------------------------------------------------------------------

    def _parse_xpu_smi_m(self) -> Optional[List[Dict[str, Any]]]:
        """调用 xpu-smi -m 并解析 machine-readable 输出。

        每行格式（空格分隔）：
            pci_addr board_id dev_id sn temp ... mem_used mem_total ...
        其中 mem_total 在第 19 个字段（0-indexed: 18）。
        """
        try:
            output = subprocess.check_output(
                ["xpu-smi", "-m"],
                text=True, timeout=15, stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

        xpus: List[Dict[str, Any]] = []
        for line in output.strip().splitlines():
            parts = line.strip().split()
            if len(parts) <= self._MR_MEM_TOTAL_POS:
                continue

            # mem_total at position 18 (MB)
            try:
                mem_total_mb = int(parts[self._MR_MEM_TOTAL_POS])
            except (ValueError, IndexError):
                mem_total_mb = 0

            # 从引号中提取产品名（如 "P800 OAM"）
            name = "Kunlun XPU"
            name_match = re.search(r'"([^"]+)"', line)
            if name_match:
                name = name_match.group(1)

            xpus.append({
                "index": len(xpus),
                "name": name,
                "memory_total_gb": round(mem_total_mb / 1024, 2),
            })

        return self._normalize_xpus(xpus)

    # ------------------------------------------------------------------
    # xpu-smi（无参数）：摘要表格格式
    # ------------------------------------------------------------------

    def _parse_xpu_smi_summary(self) -> Optional[List[Dict[str, Any]]]:
        """调用 xpu-smi（无参数）并解析摘要表格。

        典型输出：
            +------+------------+ ...
            | XPU  Name        | ...
            +------+------------+ ...
            |   0  P800 OAM    | ...
            | N/A   41C  ...   |  84504MiB / 98304MiB | ...
        """
        try:
            output = subprocess.check_output(
                ["xpu-smi"],
                text=True, timeout=15, stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        return self._parse_summary_output(output)

    def _parse_summary_output(self, output: str) -> Optional[List[Dict[str, Any]]]:
        """解析 xpu-smi 摘要表格输出。"""
        xpus: List[Dict[str, Any]] = []
        lines = output.splitlines()

        # 查找表格行: 以 "|" 开头且第一个字段是数字
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            # 跳过表头/分隔行
            if not stripped[1:].strip() or stripped.startswith("|=="):
                continue

            parts = [p.strip() for p in stripped.split("|")]
            # 期望至少: | XPU_ID | Name | ... |  Used/Total | ... |
            if len(parts) < 3:
                continue

            # 第一列数字为 XPU ID
            first_col = parts[1]
            if not first_col.isdigit():
                continue

            xpu_id = int(first_col)
            name = parts[2] if len(parts) > 2 else "Kunlun XPU"

            # 显存在 memory 列: "84504MiB / 98304MiB" 形式
            memory_gb = 0.0
            for part in parts:
                mem_match = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", part)
                if mem_match:
                    total_mb = int(mem_match.group(2))
                    memory_gb = round(total_mb / 1024, 2)
                    break

            xpus.append({
                "index": xpu_id,
                "name": name,
                "memory_total_gb": memory_gb,
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
            "98304 MiB"     -> 96.0
        """
        value = value.strip()
        match = re.search(r"([\d.]+)\s*(MB|GB|MiB|GiB|KB|KiB)?", value, re.IGNORECASE)
        if match:
            num = float(match.group(1))
            unit = (match.group(2) or "MB").upper()
            if unit in ("GB", "GIB"):
                return round(num, 2)
            if unit in ("KB", "KIB"):
                return round(num / (1024 * 1024), 2)
            # MB/MiB 都按 1024 换算
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
