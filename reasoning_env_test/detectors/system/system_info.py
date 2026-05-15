"""
系统详情检测模块

检测内容:
  - 操作系统发行版、内核版本、主机名、架构
  - CPU 详细信息（型号、核心数、缓存、频率、指令集）
  - 内存信息（总量、类型、频率、模块数）

检测方法:
  - OS: /etc/os-release, uname, platform, socket
  - CPU: /proc/cpuinfo, /sys/devices/system/cpu/
  - 内存: /proc/meminfo, dmidecode -t memory（可选，需 root）
"""

import os
import platform
import re
import subprocess
import socket
from typing import Any, Dict, List, Optional


# ============================================================
# OS 检测
# ============================================================


def detect_os() -> Dict[str, str]:
    """检测操作系统信息。

    Returns:
        dict: {
            "distribution": "Ubuntu" | "Unknown",
            "version": "22.04" | "Unknown",
            "kernel": "5.15.0-91-generic" | "Unknown",
            "hostname": "server-01" | "Unknown",
            "architecture": "x86_64" | "Unknown",
        }
    """
    result: Dict[str, str] = {
        "distribution": "Unknown",
        "version": "Unknown",
        "kernel": "Unknown",
        "hostname": "Unknown",
        "architecture": "Unknown",
    }

    # hostname
    try:
        hostname = socket.gethostname()
        if hostname:
            result["hostname"] = hostname
    except Exception:
        pass

    # architecture
    try:
        arch = platform.machine()
        if arch:
            result["architecture"] = arch
    except Exception:
        pass

    # kernel version via uname
    try:
        release = platform.uname().release
        if release:
            result["kernel"] = release
    except Exception:
        pass

    # OS distribution via /etc/os-release
    try:
        is_linux = platform.system() == "Linux"
    except Exception:
        is_linux = False

    if is_linux:
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ID="):
                        val = _parse_os_release_value(line.split("=", 1)[1])
                        if val:
                            result["distribution"] = val
                    elif line.startswith("VERSION_ID="):
                        val = _parse_os_release_value(line.split("=", 1)[1])
                        if val:
                            result["version"] = val
        except (FileNotFoundError, IOError):
            pass

    return result


def _parse_os_release_value(value: str) -> str:
    """去除 os-release 值的引号。"""
    return value.strip().strip("\"'").strip()


# ============================================================
# CPU 检测
# ============================================================


def detect_cpu() -> Dict[str, Any]:
    """检测 CPU 详细信息。

    优先从 /proc/cpuinfo 读取（所有 Linux 都有），
    缓存信息降级使用 /sys/devices/system/cpu/。

    Returns:
        dict: {
            "model_name": "Intel(R) Xeon(R) Platinum 8468",
            "architecture": "x86_64",
            "physical_cores": 96,
            "logical_cores": 192,
            "frequency_mhz": 3000.0,
            "max_frequency_mhz": 3800.0,
            "cache_l1d": "48K",
            "cache_l1i": "32K",
            "cache_l2": "2M",
            "cache_l3": "120M",
            "threads_per_core": 2,
            "sockets": 2,
            "flags": ["avx512", "avx2", ...],
        }
    """
    result: Dict[str, Any] = {
        "model_name": "Unknown",
        "architecture": "Unknown",
        "physical_cores": 0,
        "logical_cores": 0,
        "frequency_mhz": 0.0,
        "max_frequency_mhz": 0.0,
        "cache_l1d": "Unknown",
        "cache_l1i": "Unknown",
        "cache_l2": "Unknown",
        "cache_l3": "Unknown",
        "threads_per_core": 0,
        "sockets": 0,
        "flags": [],
    }

    # architecture
    try:
        arch = platform.machine()
        if arch:
            result["architecture"] = arch
    except Exception:
        pass

    # 非 Linux 系统无法读取 /proc/cpuinfo，直接返回默认值
    if platform.system() != "Linux":
        return result

    processors = _parse_cpuinfo()
    if not processors:
        return result

    first = processors[0]

    # model name
    model = first.get("model name", "").strip()
    if model:
        result["model_name"] = model

    # physical cores (from cpu cores field)
    cpu_cores_str = first.get("cpu cores", "")
    if cpu_cores_str:
        try:
            cores_per_socket = int(cpu_cores_str)
            # Get socket count from unique physical ids
            phys_ids = _get_unique_phys_ids(processors)
            socket_count = len(phys_ids) if phys_ids else 1
            result["physical_cores"] = cores_per_socket * socket_count
            result["sockets"] = socket_count
        except (ValueError, TypeError):
            pass

    # Fallback: count unique (physical_id, core_id) pairs
    if result["physical_cores"] == 0:
        core_pairs = _get_unique_core_pairs(processors)
        result["physical_cores"] = len(core_pairs)
        phys_ids = _get_unique_phys_ids(processors)
        result["sockets"] = len(phys_ids) if phys_ids else 1

    # logical cores = number of processor entries
    result["logical_cores"] = len(processors)

    # threads per core
    if result["physical_cores"] > 0:
        result["threads_per_core"] = (
            result["logical_cores"] // result["physical_cores"]
        )

    # frequency from first processor
    freq_str = first.get("cpu MHz", "")
    if freq_str:
        try:
            result["frequency_mhz"] = round(float(freq_str), 2)
        except (ValueError, TypeError):
            pass

    # max frequency
    result["max_frequency_mhz"] = _detect_max_frequency(processors)

    # cache info
    caches = _detect_cache_info(processors)
    result.update(caches)

    # flags (instruction sets)
    flags_str = first.get("flags", "")
    if isinstance(flags_str, str) and flags_str.strip():
        result["flags"] = flags_str.strip().split()

    return result


def _parse_cpuinfo() -> List[Dict[str, str]]:
    """解析 /proc/cpuinfo，返回每个逻辑处理器的属性字典列表。

    Returns:
        每个元素为一个处理器的 key-value 映射，空列表表示无可解析数据。
    """
    processors: List[Dict[str, str]] = []
    current: Dict[str, str] = {}

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current:
                        processors.append(current)
                        current = {}
                    continue
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    current[key] = value
        if current:
            processors.append(current)
    except (FileNotFoundError, IOError):
        pass

    return processors


def _get_unique_phys_ids(processors: List[Dict[str, str]]) -> set:
    """从处理器列表中提取唯一的 physical id 集合。"""
    phys_ids: set = set()
    for p in processors:
        pid = p.get("physical id")
        if pid is not None and pid.strip():
            phys_ids.add(pid.strip())
    return phys_ids


def _get_unique_core_pairs(processors: List[Dict[str, str]]) -> set:
    """从处理器列表中提取唯一的 (physical_id, core_id) 对。"""
    pairs: set = set()
    for p in processors:
        pid = p.get("physical id", "").strip()
        cid = p.get("core id", "").strip()
        if pid and cid:
            pairs.add((pid, cid))
    return pairs


def _detect_max_frequency(processors: List[Dict[str, str]]) -> float:
    """检测 CPU 最大频率（MHz）。"""
    # 优先从 /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
    try:
        path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
        if os.path.exists(path):
            with open(path) as f:
                val = f.read().strip()
                if val:
                    # 单位为 kHz
                    return round(int(val) / 1000, 2)
    except (FileNotFoundError, IOError, ValueError):
        pass

    # 降级：取所有 /proc/cpuinfo 中 cpu MHz 的最大值
    max_freq = 0.0
    for p in processors:
        freq_str = p.get("cpu MHz", "")
        if freq_str:
            try:
                freq = float(freq_str)
                if freq > max_freq:
                    max_freq = freq
            except (ValueError, TypeError):
                continue
    return round(max_freq, 2) if max_freq > 0 else 0.0


def _detect_cache_info(processors: List[Dict[str, str]]) -> Dict[str, str]:
    """检测 CPU 缓存信息。

    优先级：
      1. /proc/cpuinfo 中的 l1d cache / l1i cache / l2 cache / l3 cache 字段
      2. /sys/devices/system/cpu/cpu0/cache/index{idx}/ 目录
      3. /proc/cpuinfo 中的 cache size 字段（仅 L3）
    """
    caches: Dict[str, str] = {
        "cache_l1d": "Unknown",
        "cache_l1i": "Unknown",
        "cache_l2": "Unknown",
        "cache_l3": "Unknown",
    }

    if not processors:
        return caches

    first = processors[0]

    # 方法1：/proc/cpuinfo 中的 l1d/l1i/l2/l3 cache 字段（新内核）
    cache_keys = {
        "l1d cache": "cache_l1d",
        "l1i cache": "cache_l1i",
        "l2 cache": "cache_l2",
        "l3 cache": "cache_l3",
    }
    found_new_format = False
    for src_key, dst_key in cache_keys.items():
        val = first.get(src_key, "").strip()
        if val:
            caches[dst_key] = val
            found_new_format = True

    # 如果新格式已提供全部信息，直接返回
    if found_new_format:
        return caches

    # 方法2：/sys/devices/system/cpu/cpu0/cache/index{idx}/
    sys_cache_dir = "/sys/devices/system/cpu/cpu0/cache"
    if os.path.isdir(sys_cache_dir):
        try:
            indices = sorted(
                d for d in os.listdir(sys_cache_dir)
                if d.startswith("index") and os.path.isdir(os.path.join(sys_cache_dir, d))
            )
        except (FileNotFoundError, IOError):
            indices = []

        for idx in indices:
            idx_dir = os.path.join(sys_cache_dir, idx)
            try:
                with open(os.path.join(idx_dir, "type")) as ft:
                    cache_type = ft.read().strip()
                with open(os.path.join(idx_dir, "size")) as fs:
                    cache_size = fs.read().strip()
                with open(os.path.join(idx_dir, "level")) as fl:
                    cache_level = fl.read().strip()
            except (FileNotFoundError, IOError):
                continue

            mapped_key = None
            if cache_level == "1":
                if cache_type == "Data":
                    mapped_key = "cache_l1d"
                elif cache_type == "Instruction":
                    mapped_key = "cache_l1i"
                elif cache_type == "Unified" and caches.get("cache_l1d") == "Unknown":
                    mapped_key = "cache_l1d"
            elif cache_level == "2":
                mapped_key = "cache_l2"
            elif cache_level == "3":
                mapped_key = "cache_l3"

            if mapped_key and caches.get(mapped_key) == "Unknown":
                caches[mapped_key] = cache_size

    # 方法3：/proc/cpuinfo 的 cache size 字段（传统格式，通常为 L3）
    if caches.get("cache_l3") == "Unknown":
        cache_size = first.get("cache size", "").strip()
        if cache_size:
            caches["cache_l3"] = cache_size

    return caches


# ============================================================
# 内存检测
# ============================================================


def detect_memory() -> Dict[str, Any]:
    """检测内存信息。

    优先使用 dmidecode -t memory 获取详细内存类型/频率/模块数，
    降级时仅返回 /proc/meminfo 的总量。

    Returns:
        dict: {
            "total_gb": 1511.5,
            "type": "DDR5" | "Unknown",
            "speed_mhz": 4800,
            "modules": 16,
            "size_per_module_gb": 96,
        }
    """
    result: Dict[str, Any] = {
        "total_gb": 0.0,
        "type": "Unknown",
        "speed_mhz": 0,
        "modules": 0,
        "size_per_module_gb": 0,
    }

    if platform.system() != "Linux":
        return result

    # 总量：从 /proc/meminfo 获取
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    result["total_gb"] = round(kb / (1024 ** 2), 1)
                    break
    except (FileNotFoundError, IOError, ValueError):
        pass

    # 详细信息：尝试 dmidecode
    dmi_output = _run_dmidecode()
    if dmi_output:
        dmi_info = _parse_dmidecode_memory(dmi_output)
        result["type"] = dmi_info["type"]
        result["speed_mhz"] = dmi_info["speed_mhz"]
        result["modules"] = dmi_info["modules"]
        result["size_per_module_gb"] = dmi_info["size_per_module_gb"]

    # 降级：尝试通过 CPU 型号推断 DDR 代数
    if result["type"] in ("Unknown", ""):
        result["type"] = _infer_ddr_from_cpu()

    # 最终降级：给出明确提示
    if result["type"] in ("Unknown", ""):
        result["type"] = "Unknown (需 root 权限运行 dmidecode)"

    return result


def _infer_ddr_from_cpu() -> str:
    """通过 CPU 型号关键词推断内存 DDR 代数。"""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip().lower()
                    # DDR5: Intel 4th Gen+ Xeon (Sapphire Rapids, Emerald Rapids, Granite Rapids)
                    if any(kw in cpu for kw in ["sapphire rapids", "emerald rapids", "granite rapids",
                                                 "4th gen", "5th gen", "6th gen"]):
                        return "DDR5 (CPU 推断)"
                    if re.search(r'platinum\s+84', cpu) or re.search(r'gold\s+64', cpu):
                        return "DDR5 (CPU 推断)"
                    # DDR5: AMD EPYC 9004/8004 (Genoa, Bergamo, Siena, Turin)
                    if any(kw in cpu for kw in ["genoa", "bergamo", "siena", "turin",
                                                 "epyc 90", "epyc 80"]):
                        return "DDR5 (CPU 推断)"
                    # DDR4: Intel 1st-3rd Gen Xeon
                    if any(kw in cpu for kw in ["skylake", "cascade lake", "ice lake", "cooper lake"]):
                        return "DDR4 (CPU 推断)"
                    # DDR4: AMD EPYC 7001/7002/7003
                    if any(kw in cpu for kw in ["naples", "rome", "milan", "epyc 7"]):
                        return "DDR4 (CPU 推断)"
                    break
    except (FileNotFoundError, IOError):
        pass
    return "Unknown"


def _run_dmidecode() -> Optional[str]:
    """执行 dmidecode -t memory，返回输出文本。

    需要 root 权限或 sudo 免密配置。
    命令不存在或执行失败时返回 None。

    Returns:
        命令输出文本，失败返回 None。
    """
    # 检查 dmidecode 是否存在
    if shutil_which("dmidecode") is None:
        return None

    # 先尝试直接运行
    try:
        result = subprocess.run(
            ["dmidecode", "-t", "memory"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # 尝试 sudo
    try:
        result = subprocess.run(
            ["sudo", "-n", "dmidecode", "-t", "memory"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def shutil_which(cmd: str) -> Optional[str]:
    """调用 shutil.which 查找命令路径。"""
    import shutil
    try:
        return shutil.which(cmd)
    except Exception:
        return None


def _parse_dmidecode_memory(output: str) -> Dict[str, Any]:
    """解析 dmidecode -t memory 输出，提取内存模块信息。

    Args:
        output: dmidecode -t memory 的完整输出文本。

    Returns:
        dict: {
            "type": "DDR5" | "Unknown",
            "speed_mhz": 4800,
            "modules": 16,
            "size_per_module_gb": 96,
        }
    """
    info: Dict[str, Any] = {
        "type": "Unknown",
        "speed_mhz": 0,
        "modules": 0,
        "size_per_module_gb": 0,
    }

    # 按 Handle 行分割为不同 section
    sections = re.split(r"\n(?=Handle )", output)

    total_size_mb = 0
    module_count = 0

    for section in sections:
        # 只处理 Memory Device（DMI type 17）
        if "DMI type 17" not in section and "Memory Device" not in section:
            continue

        module_count += 1
        parsed: Dict[str, str] = {}
        for line in section.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                parsed[key.strip()] = value.strip()

        # Size
        size_str = parsed.get("Size", "No Module Installed")
        if size_str and size_str != "No Module Installed":
            size_lower = size_str.lower()
            try:
                if "gb" in size_lower:
                    gb_val = size_lower.replace("gb", "").strip()
                    total_size_mb += int(gb_val) * 1024
                elif "mb" in size_lower:
                    mb_val = size_lower.replace("mb", "").strip()
                    total_size_mb += int(mb_val)
            except (ValueError, TypeError):
                pass

        # Type（取第一个有效值）
        if info["type"] == "Unknown":
            mem_type = parsed.get("Type", "Unknown")
            if mem_type not in ("Unknown", "<OUT OF SPEC>"):
                info["type"] = mem_type

        # Speed
        if info["speed_mhz"] == 0:
            speed_str = parsed.get("Speed", "")
            m = re.search(r"(\d+)", speed_str)
            if m:
                info["speed_mhz"] = int(m.group(1))

    info["modules"] = module_count
    if module_count > 0 and total_size_mb > 0:
        info["size_per_module_gb"] = round(total_size_mb / module_count / 1024, 1)

    return info


# ============================================================
# 统一入口
# ============================================================


def detect_all() -> Dict[str, Any]:
    """运行所有系统检测，返回统一格式的结果。

    Returns:
        dict: {
            "os": { ... },     # detect_os() 返回值
            "cpu": { ... },    # detect_cpu() 返回值
            "memory": { ... }, # detect_memory() 返回值
        }
    """
    return {
        "os": detect_os(),
        "cpu": detect_cpu(),
        "memory": detect_memory(),
    }
