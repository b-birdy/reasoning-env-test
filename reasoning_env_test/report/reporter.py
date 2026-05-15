"""报告生成器：根据硬件检测、软件检测、容器检测、模型推荐、性能预估结果，生成完整中文Markdown报告。"""

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

# 命令缺失原因
_COMMAND_REASONS: Dict[str, str] = {
    "python3": "Python 运行时，推理脚本基础依赖",
    "python": "Python 运行时，推理脚本基础依赖",
    "pip": "Python 包管理，安装推理框架必需",
    "nvidia-smi": "NVIDIA GPU 监控与管理工具",
    "rocm-smi": "AMD GPU 监控与管理工具",
    "npu-smi": "昇腾 NPU 监控与管理工具",
    "docker": "容器化部署推荐，隔离运行环境",
    "kubectl": "Kubernetes 集群管理，大规模部署必需",
    "ollama": "本地大模型快速部署服务",
}


def _build_checklist_rows(
    hardware: List[Dict[str, Any]],
    software: Dict[str, Any],
    container: Dict[str, Any],
) -> List[List[str]]:
    """构建配置检查清单的行列表。

    每行格式: [检查项目, 检测结果, 状态, 建议]
    """
    rows: List[List[str]] = []

    # ── CPU 信息 ─────────────────────────────────────────────────
    for hw in hardware:
        if hw["type"] == "cpu":
            model = hw.get("model", "") or "未知"
            cores = hw.get("compute_units", 0)
            memory = hw["memory_total_gb"]
            details = hw.get("details", {})
            arch = details.get("architecture", "未知")

            rows.append(["CPU 型号", model, "✅ 正常", "-"])
            rows.append(["CPU 架构", arch, "✅ 正常", "-"])
            rows.append(["CPU 核心数", str(cores), "✅ 正常", "-"])
            rows.append(["系统内存", f"{memory:.1f} GB", "✅ 正常", "-"])
            break

    # ── GPU 信息 ─────────────────────────────────────────────────
    gpu_found = False
    for hw in hardware:
        if hw["type"] in _GPU_TYPES and hw.get("memory_total_gb", 0) > 0:
            gpu_found = True
            model = hw.get("model", "") or "未知"
            vram = hw["memory_total_gb"]
            details = hw.get("details", {})
            gpu_count = details.get("gpu_count", 1)
            rows.append([
                f"{hw['type'].title()} GPU 型号",
                model,
                "✅ 正常",
                f"共 {gpu_count} 张" if gpu_count > 1 else "单卡",
            ])
            rows.append([
                f"{hw['type'].title()} GPU 显存",
                f"{vram:.1f} GB",
                "✅ 正常",
                f"每卡 {vram:.0f} GB，共 {gpu_count} 卡"
                if gpu_count > 1 else "-",
            ])
    if not gpu_found:
        rows.append(["GPU 硬件", "未检测到", "⚠️ 缺失",
                      "服务器无 GPU 或驱动未安装，无法部署 GPU 推理模型"])

    # ── 命令检测 ─────────────────────────────────────────────────
    cmd_result = software.get("commands", {})
    for cmd in ["python3", "nvidia-smi", "rocm-smi", "npu-smi",
                "docker", "kubectl", "ollama", "pip"]:
        found = cmd_result.get(cmd, False)
        if found:
            rows.append([f"命令: {cmd}", "已安装", "✅ 正常", "-"])
        else:
            suggestion = _COMMAND_SUGGESTIONS.get(cmd, "参考官方文档安装")
            reason = _COMMAND_REASONS.get(cmd, "")
            rows.append([
                f"命令: {cmd}",
                "未安装",
                "⚠️ 缺失",
                f"{suggestion} — {reason}" if reason else suggestion,
            ])

    # ── 推理框架检测 ─────────────────────────────────────────────
    fw_result = software.get("frameworks", {})
    for fw_key, fw_ver in fw_result.items():
        display_name = fw_key.replace("_", "-")
        if fw_ver:
            rows.append([f"框架: {display_name}", fw_ver, "✅ 正常", "-"])
        else:
            suggestion = _FRAMEWORK_SUGGESTIONS.get(fw_key, "pip install")
            rows.append([
                f"框架: {display_name}",
                "未安装",
                "⚠️ 缺失",
                f"{suggestion} — 推理部署推荐安装",
            ])

    # ── Python 版本 ──────────────────────────────────────────────
    py_ver = software.get("python_version", "未知")
    py_ok = software.get("python_ok", False)
    if py_ok:
        rows.append(["Python 版本", py_ver, "✅ 正常", "≥ 3.8，满足要求"])
    else:
        rows.append([
            "Python 版本",
            py_ver,
            "⚠️ 缺失",
            "Python ≥ 3.8 是推理框架的基本要求，建议升级",
        ])

    # ── CUDA / ROCm ──────────────────────────────────────────────
    cuda_ver = software.get("cuda_version")
    rocm_ver = software.get("rocm_version")
    if cuda_ver:
        rows.append(["CUDA 版本", cuda_ver, "✅ 正常",
                      f"兼容 NVIDIA GPU 推理"])
    else:
        rows.append(["CUDA 版本", "未检测到", "⚠️ 缺失",
                      "安装 NVIDIA 驱动及 CUDA Toolkit 以启用 GPU 推理"])
    if rocm_ver:
        rows.append(["ROCm 版本", rocm_ver, "✅ 正常", "-"])
    else:
        rows.append(["ROCm 版本", "未检测到", "ℹ️ 不适用",
                      "仅 AMD GPU 需要"])

    # ── 容器状态 ─────────────────────────────────────────────────
    in_container = container.get("in_container", False)
    container_type = container.get("container_type")
    docker_installed = container.get("docker_installed", False)

    if in_container:
        ct = container_type or "未知"
        rows.append(["容器环境", f"是 ({ct})", "✅ 正常", "-"])
    else:
        rows.append(["容器环境", "否 (裸金属/虚拟机)", "✅ 正常", "-"])

    if docker_installed:
        rows.append(["Docker CLI", "已安装", "✅ 正常", "-"])
    else:
        rows.append(["Docker CLI", "未安装", "⚠️ 缺失",
                      "安装 Docker 以实现容器化部署"])

    return rows


def _build_recommendation_section(
    recommendations: Union[List[Dict[str, Any]], Dict[str, Any]],
) -> str:
    """构建模型推荐章节。"""
    lines: List[str] = []

    if isinstance(recommendations, dict) and recommendations.get("error"):
        # 无匹配模型
        err_msg = recommendations.get("message", "无法推荐模型")
        min_req = recommendations.get("min_required", {})

        lines.append("### 模型推荐结果")
        lines.append("")
        lines.append(f"⚠️ **{err_msg}**")
        lines.append("")

        if "gpu_required" in min_req:
            lines.append("> **最低配置建议**: 需要至少一张 GPU（NVIDIA/AMD/昇腾/昆仑芯/海光）")
            lines.append("> 并安装对应驱动程序，方可部署推理模型。")
        elif "lightest_model" in min_req:
            lightest = min_req["lightest_model"]
            total_mem = min_req.get("total_gpu_memory_gb", 0)
            lines.append("")
            lines.append(f"> 当前总显存: {total_mem:.1f} GB")
            lines.append(f"> 最轻量模型: **{lightest['model']}** "
                          f"({lightest['params_b']}B)")
            lines.append(f">   - FP16 最低需要 {lightest['min_vram_fp16']} GB")
            lines.append(f">   - INT8 最低需要 {lightest['min_vram_int8']} GB")
            lines.append(f">   - INT4 最低需要 {lightest['min_vram_int4']} GB")
            lines.append("")
            lines.append("> 建议增加 GPU 显存或使用更低精度量化。")
    else:
        # 有推荐模型
        lines.append("| 推荐模型 | 参数量 | 推荐精度 | 显存占用 | 模型类型 | ModelScope链接 |")
        lines.append("|---|---|---|---|---|---|")

        for rec in recommendations:
            model = rec.get("model", {})
            name = model.get("name", "未知")
            params = model.get("params_b", 0)
            precision = rec.get("precision", "fp16")
            model_type = model.get("type", "llm")
            model_scope_url = model.get("model_scope_url", "")

            # 计算显存占用
            vram_key = f"min_vram_{precision}_gb"
            vram = model.get(vram_key, 0)

            # 显存显示
            vram_display = f"{vram:.0f} GB"

            # 模型类型中文映射
            type_map = {"llm": "LLM", "multimodal": "多模态"}
            type_cn = type_map.get(model_type, model_type)

            # ModelScope 链接（缩短显示）
            url_display = f"[链接]({model_scope_url})" if model_scope_url else "-"

            lines.append(
                f"| {name} | {params}B | {precision} | {vram_display} "
                f"| {type_cn} | {url_display} |"
            )

    return "\n".join(lines)


def _build_performance_section(performance: List[Dict[str, Any]]) -> str:
    """构建性能预估章节。"""
    lines: List[str] = []

    if not performance:
        lines.append("⚠️ 无法进行性能预估（缺少硬件或模型信息）。")
        return "\n".join(lines)

    lines.append("| 并发数 | 吞吐(tok/s) | P50延迟(ms) | P99延迟(ms) | 最大支持并发 |")
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

    # 脚注：折损说明
    lines.append("")
    lines.append("> ⚠️ **性能折损说明**：以上性能数据为理论估算值，基于 40% 的")
    lines.append("> 理论到实际性能折损系数（即实际性能约为理论值的 60%）。")
    lines.append("> **实际性能**可能因硬件型号、驱动版本、软件栈、工作负载特征、")
    lines.append("> 批处理策略等而与估算值存在显著差异。")
    lines.append("> 本数据**不可替代**真实环境下的性能基准测试。")

    return "\n".join(lines)


def generate_report(
    hardware: List[Dict[str, Any]],
    software: Dict[str, Any],
    container: Dict[str, Any],
    recommendations: Union[List[Dict[str, Any]], Dict[str, Any]],
    performance: List[Dict[str, Any]],
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

    返回
    -------
    str
        完整的中文 Markdown 报告，包含三个章节和脚注。
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
    # 第一章：服务器配置检查清单
    # ================================================================
    lines.append("## 1. 服务器配置检查清单")
    lines.append("")
    lines.append("以下表格汇总了服务器的硬件配置、已安装软件、推理框架及容器环境状态。")
    lines.append("")

    rows = _build_checklist_rows(hardware, software, container)
    # 表头
    lines.append("| 检查项目 | 检测结果 | 状态 | 建议 |")
    lines.append("|---|---|---|---|")
    for row in rows:
        # 转义 Markdown 管道符
        escaped = [col.replace("|", "\\|") for col in row]
        lines.append(f"| {' | '.join(escaped)} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ================================================================
    # 第二章：可部署模型推荐
    # ================================================================
    lines.append("## 2. 可部署模型推荐")
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
    # 第三章：性能预估
    # ================================================================
    lines.append("## 3. 性能预估")
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
