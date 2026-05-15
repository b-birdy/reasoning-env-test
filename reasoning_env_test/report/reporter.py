"""报告生成器：根据硬件/系统/网络/软件/容器检测结果及模型推荐、性能预估，
生成专业中文 Markdown 格式的推理环境检测报告（共 7 章）。"""

from datetime import datetime
from typing import Any, Dict, List, Union

# GPU 硬件类型列表
_GPU_TYPES = {"nvidia", "amd", "ascend", "kunlunxin", "hygon"}

# 命令缺失时的安装建议
_COMMAND_SUGGESTIONS: Dict[str, str] = {
    "python3": "安装 Python 3: https://www.python.org/downloads/",
    "python": "安装 Python 3: https://www.python.org/downloads/",
    "pip": "安装 pip: python3 -m ensurepip",
    "nvidia-smi": "安装 NVIDIA 驱动: https://www.nvidia.com/drivers/",
    "rocm-smi": "安装 ROCm: https://rocm.docs.amd.com/",
    "npu-smi": "安装昇腾 NPU 驱动: https://www.hiascend.com/",
    "docker": "安装 Docker: https://docs.docker.com/engine/install/",
    "kubectl": "安装 kubectl: https://kubernetes.io/docs/tasks/tools/",
    "ollama": "安装 Ollama: https://ollama.com/download",
}

# 框架缺失时的安装建议
_FRAMEWORK_SUGGESTIONS: Dict[str, str] = {
    "vllm": "pip install vllm",
    "text_generation": "pip install text-generation",
    "ollama": "参考 https://ollama.com/download",
    "tensorrt_llm": "pip install tensorrt_llm",
    "onnxruntime": "pip install onnxruntime",
    "torch": "pip install torch",
    "tensorflow": "pip install tensorflow",
}

# 命令与用途映射
_COMMAND_PURPOSE: Dict[str, str] = {
    "python3": "Python 运行时",
    "python": "Python 运行时",
    "pip": "Python 包管理",
    "nvidia-smi": "NVIDIA GPU 监控",
    "rocm-smi": "AMD GPU 监控",
    "npu-smi": "昇腾 NPU 监控",
    "xpu-smi": "昆仑芯 XPU 监控",
    "docker": "容器化部署",
    "kubectl": "K8s 集群管理",
    "ollama": "本地模型部署",
    "ip": "网络接口管理",
    "ethtool": "网卡信息查询",
    "rdma": "RDMA 设备管理",
    "ibstat": "InfiniBand 状态",
    "lspci": "PCIe 设备列表",
    "dmidecode": "硬件信息查询",
    "lscpu": "CPU 信息查询",
    "lsblk": "块设备列表",
    "lshw": "硬件配置查询",
    "numactl": "NUMA 管理",
    "perf": "性能剖析",
}

# pipeline escape helper
def _e(s: str) -> str:
    """转义 Markdown 管道符。"""
    return s.replace("|", "\\|")

# ================================================================
# 第一章：概览
# ================================================================

def _section_overview(
    system_info: Dict[str, Any],
    hardware: List[Dict[str, Any]],
    network: Dict[str, Any],
) -> str:
    """生成概览章节 —— 关键信息速览表格。"""
    lines: List[str] = []
    lines.append("## 1. 概览")
    lines.append("")
    lines.append("以下表格汇总服务器核心配置概要。")
    lines.append("")

    rows: List[List[str]] = []

    # 主机名 & OS
    os_info = system_info.get("os", {}) if system_info else {}
    hostname = os_info.get("hostname", "-")
    distro = os_info.get("distribution", "-")
    kernel = os_info.get("kernel", "-")
    os_ver = os_info.get("version", "")
    os_display = f"{distro} {os_ver}".strip() if os_ver else distro
    rows.append(["主机名", hostname])
    rows.append(["操作系统", os_display])
    rows.append(["内核版本", kernel])

    # CPU
    cpu_info = system_info.get("cpu", {}) if system_info else {}
    cpu_model = cpu_info.get("model_name", "-")
    physical = cpu_info.get("physical_cores", 0)
    logical = cpu_info.get("logical_cores", 0)
    rows.append(["CPU 型号", cpu_model])
    rows.append(["CPU 核心数", f"{physical} 物理 / {logical} 逻辑"])

    # 内存
    mem_info = system_info.get("memory", {}) if system_info else {}
    total_gb = mem_info.get("total_gb", 0)
    if total_gb > 0:
        mem_type = mem_info.get("type", "")
        mem_speed = mem_info.get("speed_mhz", 0)
        mem_detail = f"{total_gb:.1f} GB"
        if mem_type and mem_type != "Unknown":
            mem_detail += f" ({mem_type}"
            if mem_speed:
                mem_detail += f" {mem_speed} MHz"
            mem_detail += ")"
        rows.append(["系统内存", mem_detail])
    else:
        # fallback: 从 hardware 取
        for hw in hardware:
            if hw["type"] == "cpu":
                rows.append(["系统内存", f"{hw.get('memory_total_gb', 0):.1f} GB"])
                break

    # GPU
    gpu_summary = _overview_gpu_summary(hardware)
    rows.append(["GPU/加速卡", gpu_summary])

    # 网络
    net_summary = network.get("summary", {}) if network else {}
    total_ports = network.get("total_network_ports", 0) if network else 0
    total_bw = net_summary.get("total_bandwidth_gbps", 0)
    net_parts = []
    if total_ports > 0:
        net_parts.append(f"{total_ports} 口")
        for label in ("400g", "200g", "100g", "25g"):
            cnt = net_summary.get(f"{label}_ports", 0)
            if cnt > 0:
                net_parts.append(f"{cnt}×{label}")
        net_parts.append(f"总带宽 {total_bw} Gbps")
    net_display = "、".join(net_parts) if net_parts else "-"
    rows.append(["网络", net_display])

    # 渲染表格
    lines.append("| 项目 | 内容 |")
    lines.append("|---|---|")
    for row in rows:
        lines.append(f"| {_e(row[0])} | {_e(row[1])} |")

    lines.append("")
    return "\n".join(lines)


def _overview_gpu_summary(hardware: List[Dict[str, Any]]) -> str:
    """从 hardware 提取 GPU 概要描述。"""
    parts: List[str] = []
    for hw in hardware:
        if hw["type"] in _GPU_TYPES and hw.get("memory_total_gb", 0) > 0:
            model = hw.get("model", "") or hw["type"]
            details = hw.get("details", {})
            count = details.get("gpu_count", 0) or details.get("xpu_count", 0) or 1
            vram = hw["memory_total_gb"]
            parts.append(f"{count}×{model} ({vram:.0f} GB /卡)")
    if not parts:
        return "未检测到 GPU"
    return "；".join(parts)


# ================================================================
# 第二章：系统信息
# ================================================================

def _section_system(system_info: Dict[str, Any]) -> str:
    """生成系统信息章节（OS / CPU / 内存）。"""
    lines: List[str] = []
    lines.append("## 2. 系统信息")
    lines.append("")

    if not system_info:
        lines.append("系统信息检测未执行或无数据。")
        lines.append("")
        return "\n".join(lines)

    # 2.1 操作系统
    os_info = system_info.get("os", {})
    lines.append("### 2.1 操作系统")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("|---|---|")
    for k, v in [("发行版", os_info.get("distribution", "Unknown")),
                  ("版本", os_info.get("version", "Unknown")),
                  ("内核", os_info.get("kernel", "Unknown")),
                  ("主机名", os_info.get("hostname", "Unknown")),
                  ("架构", os_info.get("architecture", "Unknown"))]:
        lines.append(f"| {k} | {_e(v)} |")
    lines.append("")

    # 2.2 CPU 详情
    cpu_info = system_info.get("cpu", {})
    lines.append("### 2.2 CPU 详情")
    lines.append("")
    cpu_rows = []
    cpu_rows.append(("型号", cpu_info.get("model_name", "Unknown")))
    cpu_rows.append(("架构", cpu_info.get("architecture", "Unknown")))
    cpu_rows.append(("物理核心", str(cpu_info.get("physical_cores", 0))))
    cpu_rows.append(("逻辑核心", str(cpu_info.get("logical_cores", 0))))
    threads = cpu_info.get("threads_per_core", 0)
    if threads:
        cpu_rows.append(("超线程", f"{threads} 线程/核心"))
    freq = cpu_info.get("frequency_mhz", 0)
    if freq:
        cpu_rows.append(("当前频率", f"{freq:.0f} MHz"))
    max_freq = cpu_info.get("max_frequency_mhz", 0)
    if max_freq:
        cpu_rows.append(("最大频率", f"{max_freq:.0f} MHz"))
    cpu_rows.append(("CPU 插槽", str(cpu_info.get("sockets", 0))))
    for cache_key, label in [("cache_l1d", "L1 数据缓存"),
                              ("cache_l1i", "L1 指令缓存"),
                              ("cache_l2", "L2 缓存"),
                              ("cache_l3", "L3 缓存")]:
        val = cpu_info.get(cache_key, "Unknown")
        if val and val != "Unknown":
            cpu_rows.append((label, str(val)))

    # 指令集（仅显示关键）
    important_flags = {"avx512", "avx2", "avx", "sse4_2", "sse4_1",
                       "neon", "sve", "aes", "fma3", "fma4", "amx"}
    all_flags = cpu_info.get("flags", [])
    key_flags = [f for f in all_flags if f.lower() in important_flags]
    if key_flags:
        cpu_rows.append(("关键指令集", " ".join(key_flags)))

    lines.append("| 项目 | 内容 |")
    lines.append("|---|---|")
    for k, v in cpu_rows:
        lines.append(f"| {k} | {_e(v)} |")
    lines.append("")

    # 2.3 内存详情
    mem_info = system_info.get("memory", {})
    lines.append("### 2.3 内存详情")
    lines.append("")
    mem_rows = []
    total = mem_info.get("total_gb", 0)
    mem_rows.append(("总量", f"{total:.1f} GB" if total > 0 else "Unknown"))
    mem_type = mem_info.get("type", "Unknown")
    mem_rows.append(("类型", mem_type))
    speed = mem_info.get("speed_mhz", 0)
    if speed:
        mem_rows.append(("频率", f"{speed} MHz"))
    modules = mem_info.get("modules", 0)
    if modules:
        mem_rows.append(("模块数", str(modules)))
    per_mod = mem_info.get("size_per_module_gb", 0)
    if per_mod:
        mem_rows.append(("单模块容量", f"{per_mod:.0f} GB"))

    lines.append("| 项目 | 内容 |")
    lines.append("|---|---|")
    for k, v in mem_rows:
        lines.append(f"| {k} | {_e(v)} |")
    lines.append("")

    return "\n".join(lines)


# ================================================================
# 第三章：GPU/加速卡
# ================================================================

def _section_gpu(hardware: List[Dict[str, Any]]) -> str:
    """生成 GPU/加速卡章节。"""
    lines: List[str] = []
    lines.append("## 3. GPU/加速卡")
    lines.append("")

    gpu_hw = [h for h in hardware
              if h["type"] in _GPU_TYPES and h.get("memory_total_gb", 0) > 0]

    if not gpu_hw:
        lines.append("⚠️ **未检测到 GPU 或加速卡。**")
        lines.append("")
        lines.append("> 可能原因：无 GPU 硬件、驱动未安装、或当前系统不支持此类型设备。")
        lines.append("> 建议安装对应驱动后再试。")
        lines.append("")
        return "\n".join(lines)

    total_vram = 0.0
    total_gpus = 0

    for hw in gpu_hw:
        type_cn = hw["type"].title()
        model = hw.get("model", "") or type_cn
        details = hw.get("details", {})
        gpu_count = details.get("gpu_count", 0) or details.get("xpu_count", 0) or 1
        vram = hw["memory_total_gb"]
        total_vram += vram * gpu_count
        total_gpus += gpu_count

        lines.append(f"### 3.{gpu_hw.index(hw) + 1} {type_cn}")
        lines.append("")
        lines.append(f"- **型号**: {model}")
        lines.append(f"- **数量**: {gpu_count} 张")
        lines.append(f"- **总显存**: {vram:.1f} GB（单卡）")

        # 逐卡详情（如有）
        xpus = details.get("xpus", [])
        if xpus:
            lines.append("")
            lines.append("| 设备索引 | 型号 | 显存 (GB) |")
            lines.append("|---|---|---|")
            for xpu in xpus:
                idx = xpu.get("index", "-")
                name = xpu.get("name", model)
                mem = xpu.get("memory_total_gb", 0)
                lines.append(f"| {idx} | {_e(name)} | {mem:.1f} |")

        lines.append("")

    # 汇总
    lines.append(f"> **GPU 汇总**: 共 **{total_gpus}** 张加速卡，总计 **{total_vram:.0f} GB** 显存。")
    lines.append("")

    return "\n".join(lines)


# ================================================================
# 第四章：网络
# ================================================================

def _section_network(network: Dict[str, Any]) -> str:
    """生成网络检测章节。"""
    lines: List[str] = []
    lines.append("## 4. 网络")
    lines.append("")

    if not network or not network.get("interfaces"):
        lines.append("网络检测未执行或无数据（非 Linux 系统或无 ip 命令）。")
        lines.append("")
        return "\n".join(lines)

    interfaces = network.get("interfaces", [])
    rdma_devices = network.get("rdma_devices", [])
    summary = network.get("summary", {})

    # 4.1 接口列表
    lines.append("### 4.1 网络接口")
    lines.append("")
    lines.append("| 接口名 | 类型 | 速率 | 状态 | MAC 地址 | RDMA |")
    lines.append("|---|---|---|---|---|---|")
    for iface in interfaces:
        name = iface.get("name", "-")
        iface_type = iface.get("type", "-")
        speed = iface.get("speed_gbps", 0)
        speed_str = f"{speed} Gbps" if speed > 0 else "-"
        state = iface.get("state", "unknown")
        mac = iface.get("mac", "-")
        rdma = "✅ 是" if iface.get("rdma") else "-"
        lines.append(f"| {name} | {iface_type} | {speed_str} | {state} | {mac} | {rdma} |")
    lines.append("")

    # 4.2 汇总
    lines.append("### 4.2 网络汇总")
    lines.append("")
    lines.append(f"- **接口总数**: {network.get('total_network_ports', 0)}")
    for label in ("400g", "200g", "100g", "25g", "10g"):
        cnt = summary.get(f"{label}_ports", 0)
        if cnt > 0:
            lines.append(f"- **{label.upper()} 端口**: {cnt}")
    total_bw = summary.get("total_bandwidth_gbps", 0)
    if total_bw:
        lines.append(f"- **总带宽**: {total_bw} Gbps")

    if rdma_devices:
        lines.append(f"- **RDMA 设备**: {', '.join(rdma_devices)}")
        lines.append("")
        lines.append("> ✅ 检测到 RDMA 设备 — 适合高性能分布式推理部署。")

    lines.append("")
    return "\n".join(lines)


# ================================================================
# 第五章：软件环境
# ================================================================

def _section_software(software: Dict[str, Any]) -> str:
    """生成软件环境章节。"""
    lines: List[str] = []
    lines.append("## 5. 软件环境")
    lines.append("")

    # 5.1 Python
    py_ver = software.get("python_version", "Unknown")
    py_ok = software.get("python_ok", False)
    lines.append("### 5.1 Python 环境")
    lines.append("")
    status = "✅ 满足" if py_ok else "⚠️ 不满足"
    lines.append(f"- **版本**: {py_ver}  {status}（要求 ≥ 3.8）")
    lines.append("")

    # 5.2 推理框架
    lines.append("### 5.2 推理框架")
    lines.append("")
    fw_result = software.get("frameworks", {})
    if fw_result:
        lines.append("| 框架 | 版本 | 状态 | 安装建议 |")
        lines.append("|---|---|---|---|")
        for fw_key, fw_ver in fw_result.items():
            display_name = fw_key.replace("_", "-")
            if fw_ver:
                lines.append(f"| {display_name} | {_e(fw_ver)} | ✅ 已安装 | - |")
            else:
                suggestion = _FRAMEWORK_SUGGESTIONS.get(fw_key, "pip install")
                lines.append(f"| {display_name} | - | ⚠️ 未安装 | {suggestion} |")
    else:
        lines.append("⚠️ 未检测到已安装的推理框架。")
    lines.append("")

    # 5.3 CUDA / ROCm
    lines.append("### 5.3 CUDA / ROCm")
    lines.append("")
    cuda_ver = software.get("cuda_version")
    rocm_ver = software.get("rocm_version")
    if cuda_ver:
        lines.append(f"- **CUDA**: ✅ {cuda_ver}")
    else:
        lines.append("- **CUDA**: ⚠️ 未检测到（仅 NVIDIA GPU 需要）")
    if rocm_ver:
        lines.append(f"- **ROCm**: ✅ {rocm_ver}")
    else:
        lines.append("- **ROCm**: ⚠️ 未检测到（仅 AMD GPU 需要）")
    lines.append("")

    # 5.4 硬件管理工具
    hw_tools = software.get("hardware_tools", {})
    all_cmds = software.get("commands", {})
    lines.append("### 5.4 硬件管理工具")
    lines.append("")
    if hw_tools:
        lines.append("| 工具 | 用途 | 状态 |")
        lines.append("|---|---|---|")
        for tool, found in sorted(hw_tools.items()):
            purpose = _COMMAND_PURPOSE.get(tool, "-")
            status_icon = "✅" if found else "⚠️"
            status_text = "已安装" if found else "未安装"
            lines.append(f"| {tool} | {purpose} | {status_icon} {status_text} |")
    else:
        lines.append("| 工具 | 状态 |")
        lines.append("|---|---|")
        # fallback: 显示常用命令
        for cmd in ["python3", "nvidia-smi", "rocm-smi", "npu-smi", "xpu-smi",
                    "docker", "kubectl", "lspci", "dmidecode", "ip", "ethtool",
                    "rdma", "ibstat"]:
            found = all_cmds.get(cmd, False)
            status_icon = "✅" if found else "⚠️"
            status_text = "已安装" if found else "未安装"
            purpose = _COMMAND_PURPOSE.get(cmd, "-")
            lines.append(f"| {cmd} | {purpose} | {status_icon} {status_text} |")
    lines.append("")

    return "\n".join(lines)


# ================================================================
# 第六章：模型推荐（改进现有）
# ================================================================

def _section_recommendation(
    recommendations: Union[List[Dict[str, Any]], Dict[str, Any]],
) -> str:
    """生成模型推荐章节（改进现有逻辑）。"""
    lines: List[str] = []
    lines.append("## 6. 可部署模型推荐")
    lines.append("")

    if isinstance(recommendations, dict) and recommendations.get("error"):
        err_msg = recommendations.get("message", "无法推荐模型")
        min_req = recommendations.get("min_required", {})

        lines.append(f"⚠️ **{err_msg}**")
        lines.append("")

        if "gpu_required" in min_req:
            lines.append("> 需要至少一张 GPU（NVIDIA/AMD/昇腾/昆仑芯/海光）")
            lines.append("> 并安装对应驱动程序，方可部署推理模型。")
        elif "lightest_model" in min_req:
            lightest = min_req["lightest_model"]
            total_mem = min_req.get("total_gpu_memory_gb", 0)
            lines.append(f"> 当前总显存: {total_mem:.1f} GB")
            lines.append(f"> 最轻量模型: **{lightest['model']}** ({lightest['params_b']}B)")
            lines.append(f">   - FP16 最低 {lightest['min_vram_fp16']} GB")
            lines.append(f">   - INT8 最低 {lightest['min_vram_int8']} GB")
            lines.append(f">   - INT4 最低 {lightest['min_vram_int4']} GB")
            lines.append("")
            lines.append("> 建议增加 GPU 显存或使用更低精度量化。")
        lines.append("")
        return "\n".join(lines)

    # 有推荐模型
    lines.append("根据服务器显存配置，以下模型可在当前环境中部署运行（按推荐优先级排列）。")
    lines.append("")

    lines.append("| 推荐模型 | 参数量 | 推荐精度 | 显存占用 | 模型类型 | 推理框架建议 | ModelScope |")
    lines.append("|---|---|---|---|---|---|---|")

    precision_framework_map = {
        "fp16": "vLLM / TGI",
        "int8": "vLLM + AWQ / GPTQ",
        "int4": "vLLM + AWQ / GPTQ",
        "fp32": "原生 PyTorch",
    }

    for rec in recommendations:
        model = rec.get("model", {})
        name = model.get("name", "未知")
        params = model.get("params_b", 0)
        precision = rec.get("precision", "fp16")
        model_type = model.get("type", "llm")
        model_scope_url = model.get("model_scope_url", "")

        vram_key = f"min_vram_{precision}_gb"
        vram = model.get(vram_key, 0)
        vram_display = f"{vram:.0f} GB" if vram else "-"

        type_map = {"llm": "LLM", "multimodal": "多模态"}
        type_cn = type_map.get(model_type, model_type)

        fw_suggestion = precision_framework_map.get(precision, "vLLM")
        url_display = f"[链接]({model_scope_url})" if model_scope_url else "-"

        lines.append(
            f"| {name} | {params}B | {precision} | {vram_display} "
            f"| {type_cn} | {fw_suggestion} | {url_display} |"
        )

    lines.append("")
    return "\n".join(lines)


# ================================================================
# 第七章：性能预估
# ================================================================

def _section_performance(performance: List[Dict[str, Any]]) -> str:
    """生成性能预估章节（保留现有逻辑 + 格式改进）。"""
    lines: List[str] = []
    lines.append("## 7. 性能预估")
    lines.append("")

    if not performance:
        lines.append("⚠️ 无法进行性能预估（缺少硬件或模型信息）。")
        lines.append("")
        return "\n".join(lines)

    lines.append("以下为不同并发级别下的理论性能估算值（基于推荐模型）。")
    lines.append("")

    lines.append("| 并发数 | 吞吐 (tok/s) | P50 延迟 (ms) | P99 延迟 (ms) | 最大支持并发 |")
    lines.append("|---|---|---|---|---|")

    for row in performance:
        concurrency = row.get("concurrency", 0)
        tok = row.get("tok_per_sec", 0)
        p50 = row.get("ttft_p50_ms", 0)
        p99 = row.get("ttft_p99_ms", 0)
        max_sup = row.get("max_supported", 0)
        lines.append(
            f"| {concurrency} | {tok:.1f} | {p50:.1f} | {p99:.1f} | {max_sup} |"
        )

    lines.append("")
    lines.append("> ⚠️ **性能折损说明**：以上数据为理论估算值，基于 40% 的")
    lines.append("> 理论到实际性能折损系数（即实际性能约为理论值的 60%）。")
    lines.append("> **实际性能**可能因硬件型号、驱动版本、软件栈、工作负载特征、")
    lines.append("> 批处理策略等而与估算值存在显著差异。")
    lines.append("> 本数据**不可替代**真实环境下的性能基准测试。")
    lines.append("")

    return "\n".join(lines)


# ================================================================
# 主入口
# ================================================================

# GPU 类型到中文名称映射
_GPU_TYPE_NAMES = {
    "nvidia": "NVIDIA",
    "amd": "AMD",
    "ascend": "昇腾",
    "kunlunxin": "昆仑芯",
    "hygon": "海光",
}


def _build_overview_section(
    hardware: List[Dict[str, Any]],
    system_info: Dict[str, Any] = None,
    network: Dict[str, Any] = None,
) -> str:
    """构建第一章：概览。"""
    lines: List[str] = []

    # 提取信息
    hostname = "未知"
    os_dist = "未知"
    os_ver = "未知"
    cpu_model = "未知"
    cpu_cores = 0
    total_memory = 0
    gpu_info = "无"
    total_bandwidth = 0

    # 从 system_info 获取
    if system_info:
        os_info = system_info.get("os", {})
        hostname = os_info.get("hostname", "未知")
        os_dist = os_info.get("distribution", "未知")
        os_ver = os_info.get("version", "未知")

        cpu_info = system_info.get("cpu", {})
        if cpu_info.get("model_name") and cpu_info.get("model_name") != "Unknown":
            cpu_model = cpu_info["model_name"]
        if cpu_info.get("physical_cores", 0) > 0:
            cpu_cores = cpu_info["physical_cores"]

        mem_info = system_info.get("memory", {})
        if mem_info.get("total_gb", 0) > 0:
            total_memory = mem_info["total_gb"]

    # 从 hardware 补充获取
    for hw in hardware:
        if hw["type"] == "cpu":
            if cpu_model == "未知":
                cpu_model = hw.get("model", "未知")
            if cpu_cores == 0:
                cpu_cores = hw.get("compute_units", 0)
            if total_memory == 0:
                total_memory = hw.get("memory_total_gb", 0)
        elif hw["type"] in _GPU_TYPES and hw.get("memory_total_gb", 0) > 0:
            model = hw.get("model", "未知")
            vram = hw["memory_total_gb"]
            details = hw.get("details", {})
            gpu_count = details.get("gpu_count", 1)
            gpu_info = f"{_GPU_TYPE_NAMES.get(hw['type'], hw['type'])} {model} × {gpu_count} ({vram:.1f} GB)"

    # 从 network 获取
    if network:
        summary = network.get("summary", {})
        total_bandwidth = summary.get("total_bandwidth_gbps", 0)

    # 构建表格
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 主机名 | **{hostname}** |")
    lines.append(f"| 操作系统 | **{os_dist} {os_ver}** |")
    lines.append(f"| CPU 型号 | **{cpu_model}** |")
    lines.append(f"| CPU 核心数 | **{cpu_cores}** |")
    lines.append(f"| 总内存 | **{total_memory:.1f} GB** |")
    lines.append(f"| GPU 配置 | **{gpu_info}** |")
    lines.append(f"| 网络总带宽 | **{total_bandwidth} Gbps** |")

    return "\n".join(lines)


def _build_system_section(system_info: Dict[str, Any] = None) -> str:
    """构建第二章：系统信息。"""
    lines: List[str] = []

    # 2.1 操作系统
    lines.append("### 2.1 操作系统")
    lines.append("")
    if system_info and system_info.get("os"):
        os_info = system_info["os"]
        lines.append("| 项目 | 值 |")
        lines.append("|---|---|")
        dist = os_info.get("distribution", "未知")
        lines.append(f"| 发行版 | {dist} |")
        ver = os_info.get("version", "未知")
        lines.append(f"| 版本 | {ver} |")
        kernel = os_info.get("kernel", "未知")
        lines.append(f"| 内核 | {kernel} |")
        arch = os_info.get("architecture", "未知")
        lines.append(f"| 架构 | {arch} |")
    else:
        lines.append("ℹ️ 操作系统信息暂不可用。")
    lines.append("")

    # 2.2 CPU 详情
    lines.append("### 2.2 CPU 详情")
    lines.append("")
    if system_info and system_info.get("cpu"):
        cpu_info = system_info["cpu"]
        lines.append("| 项目 | 值 |")
        lines.append("|---|---|")
        model = cpu_info.get("model_name", "未知")
        lines.append(f"| 型号 | {model} |")
        arch = cpu_info.get("architecture", "未知")
        lines.append(f"| 架构 | {arch} |")
        phys_cores = cpu_info.get("physical_cores", 0)
        lines.append(f"| 物理核心 | {phys_cores} |")
        log_cores = cpu_info.get("logical_cores", 0)
        lines.append(f"| 逻辑核心 | {log_cores} |")
        sockets = cpu_info.get("sockets", 0)
        lines.append(f"| CPU 插槽 | {sockets} |")
        freq_mhz = cpu_info.get("frequency_mhz", 0.0)
        lines.append(f"| 当前频率 | {freq_mhz:.0f} MHz |")
        max_freq_mhz = cpu_info.get("max_frequency_mhz", 0.0)
        lines.append(f"| 最大频率 | {max_freq_mhz:.0f} MHz |")
        l1d = cpu_info.get("cache_l1d", "未知")
        lines.append(f"| L1d 缓存 | {l1d} |")
        l1i = cpu_info.get("cache_l1i", "未知")
        lines.append(f"| L1i 缓存 | {l1i} |")
        l2 = cpu_info.get("cache_l2", "未知")
        lines.append(f"| L2 缓存 | {l2} |")
        l3 = cpu_info.get("cache_l3", "未知")
        lines.append(f"| L3 缓存 | {l3} |")
        # 只显示关键指令集
        flags = cpu_info.get("flags", [])
        key_flags = [f for f in ["avx2", "avx512", "sse4_2", "neon", "aes", "sse4"] if f in flags]
        lines.append(f"| 关键指令集 | {', '.join(key_flags) if key_flags else '无'} |")
    else:
        lines.append("ℹ️ CPU 详情暂不可用。")
    lines.append("")

    # 2.3 内存详情
    lines.append("### 2.3 内存详情")
    lines.append("")
    if system_info and system_info.get("memory"):
        mem_info = system_info["memory"]
        lines.append("| 项目 | 值 |")
        lines.append("|---|---|")
        total_gb = mem_info.get("total_gb", 0.0)
        lines.append(f"| 总容量 | {total_gb:.1f} GB |")
        mem_type = mem_info.get("type", "未知")
        lines.append(f"| 内存类型 | {mem_type} |")
        speed_mhz = mem_info.get("speed_mhz", 0)
        lines.append(f"| 频率 | {speed_mhz} MHz |")
        modules = mem_info.get("modules", 0)
        lines.append(f"| 模块数 | {modules} |")
        size_per_module = mem_info.get("size_per_module_gb", 0.0)
        lines.append(f"| 单模块容量 | {size_per_module:.1f} GB |")
    else:
        lines.append("ℹ️ 内存详情暂不可用。")

    return "\n".join(lines)


def _build_gpu_section(hardware: List[Dict[str, Any]]) -> str:
    """构建第三章：GPU/加速卡。"""
    lines: List[str] = []

    gpus_found = []
    for hw in hardware:
        if hw["type"] in _GPU_TYPES and hw.get("memory_total_gb", 0) > 0:
            gpus_found.append(hw)

    if gpus_found:
        # 按类型分组
        from collections import defaultdict
        gpu_groups = defaultdict(list)
        for gpu in gpus_found:
            gpu_groups[gpu["type"]].append(gpu)

        # 每种类型单独小节
        total_gpus = 0
        total_vram_gb = 0.0
        for gpu_type, gpus in gpu_groups.items():
            gpu_name = _GPU_TYPE_NAMES.get(gpu_type, gpu_type)
            lines.append(f"### {gpu_name}")
            lines.append("")
            lines.append("| 索引 | 型号 | 显存 |")
            lines.append("|---|---|---|")
            for idx, gpu in enumerate(gpus):
                model = gpu.get("model", "未知")
                vram = gpu["memory_total_gb"]
                total_gpus += 1
                total_vram_gb += vram
                details = gpu.get("details", {})
                count_in_group = details.get("gpu_count", 1)
                # 如果组中有多个，分别显示
                for i in range(count_in_group):
                    lines.append(f"| {i} | {model} | {vram:.1f} GB |")
            lines.append("")

        # 汇总信息
        lines.append(f"**总计**：共 {total_gpus} 张加速卡，总显存 {total_vram_gb:.1f} GB")
    else:
        lines.append("⚠️ 未检测到 GPU/加速卡。")
        lines.append("")
        lines.append("> **建议**：如需部署高性能推理模型，建议配置 NVIDIA/AMD/昇腾等加速卡。")

    return "\n".join(lines)


def _build_network_section(network: Dict[str, Any] = None) -> str:
    """构建第四章：网络。"""
    lines: List[str] = []

    # 4.1 接口列表
    lines.append("### 4.1 接口列表")
    lines.append("")
    if network and network.get("interfaces"):
        lines.append("| 接口名 | 类型 | 速率 | 状态 | MAC 地址 |")
        lines.append("|---|---|---|---|---|")
        for iface in network["interfaces"]:
            name = iface.get("name", "未知")
            iface_type = iface.get("type", "未知")
            speed_gbps = iface.get("speed_gbps", 0)
            state = iface.get("state", "未知")
            mac = iface.get("mac", "-")
            lines.append(f"| {name} | {iface_type} | {speed_gbps} Gbps | {state} | {mac} |")
    else:
        lines.append("ℹ️ 网络接口信息暂不可用。")
    lines.append("")

    # 4.2 汇总
    lines.append("### 4.2 汇总")
    lines.append("")
    if network:
        summary = network.get("summary", {})
        lines.append("| 项目 | 值 |")
        lines.append("|---|---|")
        total_ports = network.get("total_network_ports", 0)
        lines.append(f"| 总网络端口 | {total_ports} |")
        p400 = summary.get("400g_ports", 0)
        lines.append(f"| 400G 端口 | {p400} |")
        p200 = summary.get("200g_ports", 0)
        lines.append(f"| 200G 端口 | {p200} |")
        p25 = summary.get("25g_ports", 0)
        lines.append(f"| 25G 端口 | {p25} |")
        total_bw = summary.get("total_bandwidth_gbps", 0)
        lines.append(f"| 总带宽 | {total_bw} Gbps |")

        rdma_devices = network.get("rdma_devices", [])
        lines.append("")
        if rdma_devices:
            lines.append(f"✅ **RDMA 设备**：{', '.join(rdma_devices)}")
        else:
            lines.append("ℹ️ 未检测到 RDMA 设备。")
    else:
        lines.append("ℹ️ 网络汇总信息暂不可用。")

    return "\n".join(lines)


def _build_software_section(software: Dict[str, Any]) -> str:
    """构建第五章：软件环境。"""
    lines: List[str] = []

    # 5.1 Python 环境
    lines.append("### 5.1 Python 环境")
    lines.append("")
    py_ver = software.get("python_version", "未知")
    py_ok = software.get("python_ok", False)
    status = "✅ 满足要求" if py_ok else "⚠️ 需升级至 ≥ 3.8"
    lines.append(f"| 项目 | 值 | 状态 |")
    lines.append("|---|---|---|")
    lines.append(f"| Python 版本 | {py_ver} | {status} |")
    lines.append("")

    # 5.2 推理框架
    lines.append("### 5.2 推理框架")
    lines.append("")
    fw_result = software.get("frameworks", {})
    if fw_result:
        lines.append("| 框架 | 版本 | 状态 |")
        lines.append("|---|---|---|")
        for fw_key, fw_ver in fw_result.items():
            display_name = fw_key.replace("_", "-")
            status = "✅ 已安装" if fw_ver else "⚠️ 未安装"
            lines.append(f"| {display_name} | {fw_ver or '-'} | {status} |")
    else:
        lines.append("ℹ️ 框架信息暂不可用。")
    lines.append("")

    # 5.3 CUDA / ROCm
    lines.append("### 5.3 CUDA / ROCm")
    lines.append("")
    lines.append("| 组件 | 版本 | 状态 |")
    lines.append("|---|---|---|")
    cuda_ver = software.get("cuda_version")
    if cuda_ver:
        lines.append(f"| CUDA | {cuda_ver} | ✅ 可用 |")
    else:
        lines.append("| CUDA | - | ⚠️ 未检测到 |")

    rocm_ver = software.get("rocm_version")
    if rocm_ver:
        lines.append(f"| ROCm | {rocm_ver} | ✅ 可用 |")
    else:
        lines.append("| ROCm | - | ℹ️ 不适用 |")
    lines.append("")

    # 5.4 硬件管理工具
    lines.append("### 5.4 硬件管理工具")
    lines.append("")
    cmd_result = software.get("commands", {})
    hardware_tools = ["nvidia-smi", "rocm-smi", "npu-smi", "docker"]
    lines.append("| 工具 | 状态 |")
    lines.append("|---|---|")
    for tool in hardware_tools:
        found = cmd_result.get(tool, False)
        status = "✅ 已安装" if found else "⚠️ 未安装"
        lines.append(f"| {tool} | {status} |")

    return "\n".join(lines)


def generate_report(
    hardware: List[Dict[str, Any]],
    software: Dict[str, Any],
    container: Dict[str, Any],
    recommendations: Union[List[Dict[str, Any]], Dict[str, Any]],
    performance: List[Dict[str, Any]],
    system_info: Dict[str, Any] = None,
    network: Dict[str, Any] = None,
) -> str:
    """生成完整中文 Markdown 格式环境检测报告。

    参数
    ----------
    hardware : List[Dict]
        硬件检测结果列表（来自 hardware.detect_all()）。
    software : Dict
        软件检测结果（来自 software.detect_all()）。
    container : Dict
        容器检测结果（来自 container.detect_all()）。
    recommendations : List[Dict] or Dict
        模型推荐结果（来自 recommender.recommend()）。
    performance : List[Dict]
        性能预估结果（来自 estimator.estimate_performance()）。
    system_info : Dict, optional
        系统信息（来自 system.detect_all()），默认为 None。
    network : Dict, optional
        网络信息（来自 network.detect_all()），默认为 None。

    返回
    -------
    str
        完整的中文 Markdown 报告，包含七个章节。
    """
    lines: List[str] = []

    # ── 报告标题 ─────────────────────────────────────────────────
    lines.append("# 推理环境检测报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ================================================================
    # 第一章：概览
    # ================================================================
    lines.append("## 1. 概览")
    lines.append("")
    lines.append("关键信息速览，快速掌握服务器配置概要。")
    lines.append("")
    lines.append(_build_overview_section(hardware, system_info, network))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ================================================================
    # 第二章：系统信息
    # ================================================================
    lines.append("## 2. 系统信息")
    lines.append("")
    lines.append("详细的操作系统、CPU 和内存配置信息。")
    lines.append("")
    lines.append(_build_system_section(system_info))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ================================================================
    # 第三章：GPU/加速卡
    # ================================================================
    lines.append("## 3. GPU/加速卡")
    lines.append("")
    lines.append("服务器配置的 GPU 和 AI 加速卡信息。")
    lines.append("")
    lines.append(_build_gpu_section(hardware))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ================================================================
    # 第四章：网络
    # ================================================================
    lines.append("## 4. 网络")
    lines.append("")
    lines.append("网络接口配置和 RDMA 设备信息。")
    lines.append("")
    lines.append(_build_network_section(network))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ================================================================
    # 第五章：软件环境
    # ================================================================
    lines.append("## 5. 软件环境")
    lines.append("")
    lines.append("Python 版本、推理框架、CUDA/ROCm 及硬件管理工具。")
    lines.append("")
    lines.append(_build_software_section(software))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ================================================================
    # 第六章：可部署模型推荐
    # ================================================================
    lines.append("## 6. 可部署模型推荐")
    lines.append("")
    lines.append(
        "根据服务器显存配置，以下模型可在当前环境中部署运行"
        "（按参数量从大到小排序）。"
    )
    lines.append("")
    lines.append(_build_recommendation_section(recommendations))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ================================================================
    # 第七章：性能预估
    # ================================================================
    lines.append("## 7. 性能预估")
    lines.append("")
    lines.append("以下为各并发级别下的理论性能估算值。")
    lines.append("")
    lines.append(_build_performance_section(performance))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 报告结尾 ─────────────────────────────────────────────────
    lines.append("")
    lines.append("*报告由推理环境检测工具自动生成*")

    return "\n".join(lines)
