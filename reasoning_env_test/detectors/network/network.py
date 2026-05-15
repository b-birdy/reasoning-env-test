"""
网络接口检测模块

检测网络接口、RDMA/RoCE 设备、链路速率和接口类型。
仅 Linux 平台可正常工作；非 Linux 返回空结果。

检测方法（逐级降级）:
  1. ip -details link show — 详细接口类型和状态
  2. ip link show — 基础接口信息（降级）
  3. ethtool <iface> — 速率检测（以太口）
  4. rdma link show — RDMA 设备列表
  5. /sys/class/net/<iface>/speed — 直接读取 sysfs 速率
  6. /sys/class/net/<iface>/type — sysfs 接口类型
"""

import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Set, Tuple

# sysfs 网络接口路径
SYS_CLASS_NET = "/sys/class/net"

# 速率分类阈值 (Gbps)
SPEED_CATEGORIES: Dict[str, int] = {
    "400g": 400,
    "200g": 200,
    "100g": 100,
    "50g": 50,
    "25g": 25,
    "10g": 10,
}

# sysfs 接口类型号 -> 名称映射
SYSFS_TYPE_MAP: Dict[str, str] = {
    "1": "ethernet",
    "32": "infiniband",
    "772": "loopback",
}


def _run_cmd(cmd: List[str], timeout: int = 10) -> str | None:
    """安全运行 shell 命令，返回 stdout 或 None。"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _read_sysfs(path: str) -> str | None:
    """安全读取 sysfs 文件，返回内容或 None。"""
    try:
        if os.path.isfile(path):
            with open(path) as f:
                return f.read().strip()
    except (FileNotFoundError, PermissionError, IOError):
        pass
    return None


def _empty_result() -> Dict[str, Any]:
    """返回空检测结果。"""
    return {
        "interfaces": [],
        "rdma_devices": [],
        "total_network_ports": 0,
        "summary": {
            "400g_ports": 0,
            "200g_ports": 0,
            "25g_ports": 0,
            "total_bandwidth_gbps": 0,
        },
    }


def _speed_to_gbps(speed_val: str | int | None) -> int:
    """将速率值转换为 Gbps。

    处理:
      - sysfs 值（单位 Mbps，如 400000）
      - ethtool 格式（如 '400000Mb/s', '100Gb/s', '1Tb/s'）
    """
    if speed_val is None:
        return 0

    if isinstance(speed_val, int):
        return speed_val // 1000

    s = str(speed_val).strip().lower()

    m = re.match(r"(\d+)", s)
    if not m:
        return 0

    val = int(m.group(1))

    if "t" in s:
        return val * 1000  # Tb/s -> Gb/s
    if "g" in s:
        return val  # 已是 Gb/s
    # 默认 Mbps -> Gbps
    return val // 1000


def _classify_speed(gbps: int) -> str:
    """将 Gbps 速率归入命名类别。"""
    for category, threshold in sorted(
        SPEED_CATEGORIES.items(), key=lambda x: -x[1],
    ):
        if gbps >= threshold:
            return category
    return "other"


def _is_rdma_by_name(name: str) -> bool:
    """通过接口名判断是否为 RDMA 设备。"""
    name_lower = name.lower()
    rdma_name_patterns = ["ib", "rxe", "mlx", "roce"]
    return any(p in name_lower for p in rdma_name_patterns)


# ============================================================
# RDMA 检测
# ============================================================


def _get_rdma_info() -> Tuple[List[str], Set[str]]:
    """获取 RDMA 设备信息和 RDMA 能力网卡集合。

    Returns:
        (rdma_device_names, rdma_netdevs):
            rdma_device_names: RDMA 设备名列表（如 ["rxe_0", "mlx5_0"]）
            rdma_netdevs: 具备 RDMA 能力的网络接口名集合（如 {"eth0", "eth1"}）
    """
    rdma_devices: List[str] = []
    rdma_netdevs: Set[str] = set()

    # 方法 1: rdma link show
    output = _run_cmd(["rdma", "link", "show"])
    if output:
        for line in output.splitlines():
            # 匹配设备名: "1: rxe_0: state ACTIVE ..."
            m = re.match(r"\s*\d+:\s+(\S+?)(?:\s*$|[\s/:])", line)
            if m:
                dev_name = m.group(1).strip()
                if dev_name not in rdma_devices:
                    rdma_devices.append(dev_name)

            # 匹配关联的 netdev: "... netdev eth0"
            netdev_m = re.search(r"netdev\s+(\S+)", line)
            if netdev_m:
                rdma_netdevs.add(netdev_m.group(1))
        return rdma_devices, rdma_netdevs

    # 方法 2: 回退 sysfs
    try:
        if os.path.isdir("/sys/class/infiniband"):
            for name in os.listdir("/sys/class/infiniband"):
                if name not in rdma_devices:
                    rdma_devices.append(name)
    except (FileNotFoundError, PermissionError):
        pass

    return rdma_devices, rdma_netdevs


# ============================================================
# 速率检测
# ============================================================


def _get_speed_from_sysfs(iface: str) -> int:
    """从 sysfs 读取接口速率。"""
    speed_str = _read_sysfs(f"{SYS_CLASS_NET}/{iface}/speed")
    if speed_str is not None and speed_str != "-1":
        return _speed_to_gbps(speed_str)
    return 0


def _get_speed_from_ethtool(iface: str) -> int:
    """通过 ethtool 获取接口速率。"""
    output = _run_cmd(["ethtool", iface])
    if not output:
        return 0

    for line in output.splitlines():
        m = re.search(r"Speed:\s*(\S+)", line)
        if m:
            return _speed_to_gbps(m.group(1))
    return 0


# ============================================================
# 接口类型检测
# ============================================================


def _get_iface_type_from_sysfs(iface: str) -> str | None:
    """从 sysfs 获取接口类型名称。"""
    type_str = _read_sysfs(f"{SYS_CLASS_NET}/{iface}/type")
    if type_str is None:
        return None
    return SYSFS_TYPE_MAP.get(type_str)


def _determine_iface_type(
    name: str,
    link_type: str,
    sysfs_type: str | None,
    rdma_netdevs: Set[str],
) -> str:
    """综合判定接口类型。

    优先级: ip link_type > sysfs type > RDMA 关联 > 命名模式 > 默认 ethernet
    """
    # 链路层类型优先
    if link_type == "infiniband":
        return "infiniband"
    if link_type == "loopback":
        return "loopback"

    # sysfs 类型
    if sysfs_type == "infiniband":
        return "infiniband"

    # RDMA 关联网卡（RoCE）
    if name in rdma_netdevs:
        return "roce"

    # 命名模式
    if _is_rdma_by_name(name):
        name_lower = name.lower()
        if "ib" in name_lower:
            return "infiniband"
        return "roce"

    return "ethernet"


# ============================================================
# ip link show 输出解析
# ============================================================


def _parse_ip_link_output(output: str) -> List[Dict[str, Any]]:
    """解析 'ip link show' 输出，返回接口属性列表。

    兼容不同 iproute2 版本的输出格式差异。
    """
    interfaces: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    for line in output.splitlines():
        # ── 接口头部行 ──
        # 格式: INDEX: NAME@PARENT: <FLAGS> ... state STATE ...
        m = re.match(r"^(\d+):\s+(\S+?)(?:@\S+)?:\s+<(.+?)>", line)
        if m:
            # 保存前一个接口
            if current is not None:
                interfaces.append(current)

            current = {
                "index": int(m.group(1)),
                "name": m.group(2),
                "flags": m.group(3),
                "mac": "",
                "link_type": "",
                "state": "unknown",
            }

            # 提取状态 (state 字段可能在行中不同位置)
            state_m = re.search(r"state\s+(\S+)", line)
            if state_m:
                current["state"] = state_m.group(1).lower()

            # 一些旧版 ip 可能在头部行包含 MAC（罕见）
            mac_m = re.search(
                r"((?:[0-9a-f]{2}:){5}[0-9a-f]{2})", line, re.IGNORECASE,
            )
            if mac_m:
                current["mac"] = mac_m.group(1).lower()

            continue

        # ── 链路层信息行 ──
        # 格式: link/TYPE MAC_ADDR ...
        link_m = re.match(r"\s+link/(\S+)", line)
        if link_m and current is not None:
            current["link_type"] = link_m.group(1)

            # 提取 MAC 地址（仅以太网/loopback，避免误匹配 InfiniBand 长地址）
            if current["link_type"] in ("ether", "loopback"):
                mac_m = re.search(
                    r"((?:[0-9a-f]{2}:){5}[0-9a-f]{2})", line, re.IGNORECASE,
                )
                if mac_m:
                    current["mac"] = mac_m.group(1).lower()

    # 最后一个接口
    if current is not None:
        interfaces.append(current)

    return interfaces


# ============================================================
# 接口检测主流程
# ============================================================


def detect_interfaces() -> List[Dict[str, Any]]:
    """检测所有网络接口及其属性。

    Returns:
        接口字典列表，每个接口包含 name, type, speed_gbps, state, mac, rdma 等字段。
    """
    rdma_devices, rdma_netdevs = _get_rdma_info()

    # 优先 ip -details（包含更多信息），回退 ip link show
    output = _run_cmd(["ip", "-details", "link", "show"])
    if output is None:
        output = _run_cmd(["ip", "link", "show"])
        if output is None:
            return []

    parsed = _parse_ip_link_output(output)
    if not parsed:
        return []

    interfaces: List[Dict[str, Any]] = []

    for entry in parsed:
        name = entry["name"]
        # 跳过 loopback
        if name == "lo":
            continue

        # 判定类型
        sysfs_type = _get_iface_type_from_sysfs(name)
        iface_type = _determine_iface_type(
            name,
            entry.get("link_type", ""),
            sysfs_type,
            rdma_netdevs,
        )

        # 速率检测（对以太/ RoCE 口有效）
        speed = 0
        if iface_type in ("ethernet", "roce"):
            speed = _get_speed_from_ethtool(name)
            if speed == 0:
                speed = _get_speed_from_sysfs(name)

        # RDMA 判断
        is_rdma = (name in rdma_netdevs
                    or iface_type in ("infiniband", "roce")
                    or _is_rdma_by_name(name))

        iface_dict: Dict[str, Any] = {
            "name": name,
            "type": iface_type,
            "speed_gbps": speed,
            "state": entry.get("state", "unknown"),
        }

        if entry.get("mac"):
            iface_dict["mac"] = entry["mac"]

        if is_rdma:
            iface_dict["rdma"] = True

        interfaces.append(iface_dict)

    return interfaces


# ============================================================
# 统一入口
# ============================================================


def detect_all() -> Dict[str, Any]:
    """运行所有网络检测，返回统一格式结果。

    仅在 Linux 平台且 ip 命令可用时执行检测；
    否则返回空结果。

    Returns:
        dict: {
            "interfaces": [
                {"name": "eth0", "type": "ethernet", "speed_gbps": 25,
                 "state": "up", "mac": "xx:xx:xx:xx:xx:xx"},
                {"name": "ib0", "type": "infiniband", "speed_gbps": 400,
                 "state": "up", "rdma": True},
                {"name": "roce_0", "type": "roce", "speed_gbps": 400,
                 "state": "up", "rdma": True},
            ],
            "rdma_devices": ["rxe_0", "mlx5_0"],
            "total_network_ports": 12,
            "summary": {
                "400g_ports": 8,
                "200g_ports": 2,
                "25g_ports": 3,
                "total_bandwidth_gbps": 3600,
            },
        }
    """
    # 非 Linux 系统不支持
    if sys.platform not in ("linux", "linux2"):
        return _empty_result()

    # 无 ip 命令
    if shutil.which("ip") is None:
        return _empty_result()

    interfaces = detect_interfaces()
    rdma_devices, _ = _get_rdma_info()

    # 计算汇总
    total_bandwidth = 0
    speed_counts: Dict[str, int] = {}

    for iface in interfaces:
        gbps = iface.get("speed_gbps", 0)
        total_bandwidth += gbps
        category = _classify_speed(gbps)
        speed_counts[category] = speed_counts.get(category, 0) + 1

    summary = {
        "400g_ports": speed_counts.get("400g", 0),
        "200g_ports": speed_counts.get("200g", 0),
        "25g_ports": speed_counts.get("25g", 0),
        "total_bandwidth_gbps": total_bandwidth,
    }

    return {
        "interfaces": interfaces,
        "rdma_devices": rdma_devices,
        "total_network_ports": len(interfaces),
        "summary": summary,
    }
