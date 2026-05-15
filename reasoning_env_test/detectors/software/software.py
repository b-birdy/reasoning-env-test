"""
软件与推理框架检测模块

检测内容:
  - 扫描 BIN_DIRS + PATH 下的可执行文件
  - 常用命令检测（python, nvidia-smi, docker 等）
  - 硬件管理工具检测（xpu-smi, dmidecode, lspci 等）
  - 推理框架检测（vLLM, TGI, Ollama, PyTorch 等）
  - Python 版本检查（>= 3.8）
  - CUDA / ROCm 版本
"""

import os
import re
import shutil
import subprocess
import sys
from typing import Optional


# ============================================================
# 常量
# ============================================================

# 标准 bin 目录（仅 Linux 常用）
BIN_DIRS = ["/bin", "/usr/bin", "/usr/local/bin"]

# 待检测命令列表
COMMANDS = [
    "python3",
    "python",
    "pip",
    "nvidia-smi",
    "rocm-smi",
    "npu-smi",
    "xpu-smi",
    "hy-smi",
    "mthreads-smi",
    "cnmon",
    "biren-smi",
    "msmi",
    "smi",
    "docker",
    "kubectl",
    "ollama",
    "ip",
    "ethtool",
    "rdma",
    "ibstat",
    "lspci",
    "dmidecode",
    "lscpu",
    "lsblk",
    "lshw",
    "numactl",
    "perf",
]

# 待检测硬件管理工具（与 COMMANDS 独立，用于 hardware_tools 字段）
HARDWARE_TOOLS = [
    "xpu-smi",      # 昆仑芯
    "nvidia-smi",   # NVIDIA
    "rocm-smi",     # AMD
    "hy-smi",       # 海光 DCU
    "npu-smi",      # 昇腾
    "mthreads-smi", # 摩尔线程
    "cnmon",        # 寒武纪
    "biren-smi",    # 壁仞
    "msmi",         # 天数智芯
    "smi",          # 燧原
    "dmidecode",
    "lspci",
    "lshw",
    "lscpu",
    "ip",
    "ethtool",
    "rdma",
    "ibstat",
]

# 推理框架 pip 包名 -> 输出 key
FRAMEWORKS = {
    "vllm": "vllm",
    "sglang": "sglang",
    "text-generation": "text_generation",
    "ollama": "ollama",
    "tensorrt_llm": "tensorrt_llm",
    "onnxruntime": "onnxruntime",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "deepspeed": "deepspeed",
    "flash-attn": "flash_attn",
    "xformers": "xformers",
    "bitsandbytes": "bitsandbytes",
    "peft": "peft",
}

# 通信库 pip 包名 -> 输出 key 及别名
COMM_LIBS = {
    "nccl": "nccl",
    "torch_nccl": "torch_nccl",
    "hccl": "hccl",
    "bkcl": "bkcl",
    "xccl": "xccl",
    "msccl": "msccl",
    "rccl": "rccl",
}


# ============================================================
# bin 目录扫描
# ============================================================


def scan_bin_dirs() -> list[str]:
    """扫描常见的 bin 目录和 PATH，返回所有可执行文件列表。

    同时扫描 BIN_DIRS 和 PATH 中的所有目录，去重后返回。
    """
    found: list[str] = []
    seen: set[str] = set()
    dirs_to_scan: list[str] = list(BIN_DIRS)

    # 追加 PATH 中的目录（去重）
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for d in path_dirs:
        if d and d not in dirs_to_scan:
            dirs_to_scan.append(d)

    for d in dirs_to_scan:
        if not os.path.isdir(d):
            continue
        try:
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if fp not in seen and os.access(fp, os.X_OK):
                    found.append(fp)
                    seen.add(fp)
        except PermissionError:
            continue

    return sorted(found)


# ============================================================
# 命令检测
# ============================================================


def detect_commands() -> dict[str, bool]:
    """检测常用命令是否存在于当前环境中。"""
    result: dict[str, bool] = {}
    for cmd in COMMANDS:
        result[cmd] = shutil.which(cmd) is not None
    return result


def detect_hardware_tools() -> dict[str, bool]:
    """检测硬件管理工具是否存在于当前环境中。"""
    return {tool: shutil.which(tool) is not None for tool in HARDWARE_TOOLS}


# ============================================================
# 推理框架检测
# ============================================================


def _get_pip_packages() -> dict[str, str]:
    """获取当前环境已安装的 pip 包名 -> 版本 映射。"""
    packages: dict[str, str] = {}

    # Python >= 3.8 推荐 importlib.metadata
    try:
        from importlib.metadata import distributions

        for dist in distributions():
            name = dist.metadata.get("Name", "").lower()
            ver = dist.version or ""
            if name:
                packages[name] = ver
        return packages
    except ImportError:
        pass

    # 回退 pkg_resources（setuptools）
    try:
        import pkg_resources  # type: ignore[import-untyped]

        for pkg in pkg_resources.working_set:
            packages[pkg.key.lower()] = pkg.version
    except ImportError:
        pass

    return packages


def _get_version_via_which(name: str) -> Optional[str]:
    """尝试通过 which + --version 获取命令版本。"""
    path = shutil.which(name)
    if not path:
        return None
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = (result.stdout or result.stderr).strip()
        return out if out else None
    except Exception:
        return None


def detect_frameworks() -> dict[str, Optional[str]]:
    """检测已安装的推理框架及其版本。

    优先通过 pip 包（importlib.metadata）检测，失败时回退到 which。
    """
    pip_pkgs = _get_pip_packages()
    result: dict[str, Optional[str]] = {}

    for pkg_name, key in FRAMEWORKS.items():
        pkg_lower = pkg_name.lower()
        if pkg_lower in pip_pkgs:
            result[key] = pip_pkgs[pkg_lower]
        else:
            # 回退：尝试 which 检测
            ver = _get_version_via_which(pkg_name)
            result[key] = ver if ver else None

    return result


# ============================================================
# Python 版本检测
# ============================================================


def detect_python() -> tuple[str, bool]:
    """检测 Python 版本并判断是否 >= 3.8。"""
    info = sys.version_info
    version_str = f"{info.major}.{info.minor}.{info.micro}"
    ok = info.major >= 3 and info.minor >= 8
    return version_str, ok


# ============================================================
# CUDA 版本检测
# ============================================================


def detect_cuda() -> Optional[str]:
    """检测 CUDA 版本。

    优先级:
      1. nvidia-smi 输出中的 CUDA Version
      2. nvcc --version
    """
    # 1) nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                m = re.search(r"CUDA Version:\s*([\d.]+)", line)
                if m:
                    return m.group(1)
        except Exception:
            pass

    # 2) nvcc --version
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            result = subprocess.run(
                [nvcc, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                m = re.search(r"release\s+([\d.]+)", line)
                if m:
                    return m.group(1)
        except Exception:
            pass

    return None


# ============================================================
# ROCm 版本检测
# ============================================================


def detect_rocm() -> Optional[str]:
    """检测 ROCm 版本。

    优先级:
      1. rocm-smi --showversion
      2. /opt/rocm/version 文件
    """
    # 1) rocm-smi
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        try:
            result = subprocess.run(
                [rocm_smi, "--showversion"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                # 取第一个看起来像版本号的 token
                m = re.search(r"(\d+\.\d+\.\d+)", line)
                if m:
                    return m.group(1)
        except Exception:
            pass

    # 2) /opt/rocm/version 文件
    try:
        if os.path.exists("/opt/rocm/version"):
            with open("/opt/rocm/version") as f:
                ver = f.read().strip()
                if ver:
                    return ver
    except Exception:
        pass

    return None


# ============================================================
# Docker 版本检测
# ============================================================


def detect_docker_version() -> Optional[str]:
    """检测 Docker 版本。"""
    path = shutil.which("docker")
    if not path:
        return None
    try:
        result = subprocess.run(
            [path, "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            ver = result.stdout.strip()
            if ver:
                return ver
    except Exception:
        pass
    # 回退: docker --version
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        out = (result.stdout or result.stderr).strip()
        if out:
            # 提取版本号 "Docker version 24.0.7, build ..."
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


# ============================================================
# 通信库检测 (NCCL / HCCL / BKCL / RCCL)
# ============================================================


def detect_comm_libs() -> dict[str, Optional[str]]:
    """检测 GPU 通信库版本。

    优先通过 pip 包检测，降级到 ldconfig / 路径扫描。

    Returns:
        {"nccl": "2.20.5" | None, "hccl": ..., "bkcl": ..., "rccl": ...}
    """
    pip_pkgs = _get_pip_packages()
    result: dict[str, Optional[str]] = {
        "nccl": None,
        "hccl": None,
        "bkcl": None,
        "rccl": None,
        "xccl": None,
    }

    # 1) pip 包
    for pkg_name, key in COMM_LIBS.items():
        pkg_lower = pkg_name.lower()
        if pkg_lower in pip_pkgs:
            result[key] = pip_pkgs[pkg_lower]

    # 2) 路径 / 环境变量
    _detect_comm_lib_path("nccl", "/usr/lib/x86_64-linux-gnu/libnccl.so", result)
    _detect_comm_lib_path("hccl", "/usr/local/Ascend/ascend-toolkit/latest/atc/lib64/libhccl.so", result)
    _detect_comm_lib_path("bkcl", "/usr/lib/x86_64-linux-gnu/libbkcl.so", result)
    _detect_comm_lib_path("rccl", "/opt/rocm/lib/librccl.so", result)

    return result


def _detect_comm_lib_path(key: str, so_path: str, result: dict) -> None:
    """通过 .so 文件版本检测通信库（降级方案）。"""
    if result.get(key):
        return  # pip 已找到
    try:
        if os.path.exists(so_path):
            # readlink -f 获取真实路径
            real = os.path.realpath(so_path)
            m = re.search(r"(\d+\.\d+\.\d+)", real)
            if m:
                result[key] = m.group(1)
            else:
                result[key] = "detected"
    except Exception:
        pass


# ============================================================
# 统一入口
# ============================================================


def detect_all() -> dict:
    """运行所有检测，返回统一格式的结果。

    Returns:
        dict: {
            "commands": {"name": bool, ...},
            "frameworks": {"name": str | None, ...},
            "python_version": "x.y.z",
            "python_ok": bool,
            "cuda_version": str | None,
            "rocm_version": str | None,
            "docker_version": str | None,
            "comm_libs": {"nccl": ..., "hccl": ..., "bkcl": ..., "rccl": ...},
            "all_commands": list[str],
            "hardware_tools": {"name": bool, ...},
        }
    """
    return {
        "commands": detect_commands(),
        "frameworks": detect_frameworks(),
        "python_version": detect_python()[0],
        "python_ok": detect_python()[1],
        "cuda_version": detect_cuda(),
        "rocm_version": detect_rocm(),
        "docker_version": detect_docker_version(),
        "comm_libs": detect_comm_libs(),
        "all_commands": scan_bin_dirs(),
        "hardware_tools": detect_hardware_tools(),
    }
