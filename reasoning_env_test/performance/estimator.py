"""性能预估模块。

基于 GPU 规格信息，估算指定模型在多并发下的吞吐量（tok/s）和首包延迟（TTFT）。

⚠️ 重要提示
    所有数值均为 **理论估算值**，基于 40% 理论到实际性能折损。
    LLM 推理解码阶段主要是 **内存绑定** 而非计算绑定。
    本模块的输出 **不可替代** 真实环境下的性能基准测试。
"""
from typing import Any, Dict, List
import json
import os


# ---------------------------------------------------------------------------
# 精度字节数
# ---------------------------------------------------------------------------
_PRECISION_BYTES: Dict[str, float] = {
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# ---------------------------------------------------------------------------
# 理论到实际折损
# ---------------------------------------------------------------------------
_DEGRADATION: float = 0.6


def _get_gpu_spec_path() -> str:
    """获取 GPU 规格文件路径。"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    return os.path.join(data_dir, "gpu_specs.json")


def load_gpu_specs() -> Dict[str, Dict[str, Any]]:
    """加载 GPU 规格字典。

    返回
    -------
    Dict[str, Dict[str, Any]]
        键为 GPU 型号 ID，值为包含规格的字典。
    """
    with open(_get_gpu_spec_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_performance(
    gpu_spec: Dict[str, Any],
    model_params_b: float,
    precision: str,
    concurrency_levels: List[int],
    gpu_count: int,
    memory_bw_factor: float = 1.0,
    rdma_latency_us: float = 5.0,
    scenario: str = "single_node",
    model_name: str = "",
    deploy_desc: str = "",
) -> List[Dict[str, Any]]:
    """估算多并发下性能指标。

    参数
    ----------
    gpu_spec : Dict[str, Any]
        单张 GPU 的规格字典，来自 gpu_specs.json 查找。

    model_params_b : float
        模型参数量，以十亿（B）为单位。如 7B 则传入 7.0。

    precision : str
        推理精度。可选: "fp16", "int8", "int4"。

    concurrency_levels : List[int]
        需要估算的并发级别列表，如 [4, 8, 16, 32, 64, 128]。

    gpu_count : int
        使用的 GPU 数量。

    memory_bw_factor : float, optional
        多节点内存带宽 overhead，默认为 1.0（无 overhead）。

    rdma_latency_us : float, optional
        RDMA 往返延迟（微秒），默认为 5.0。

    scenario : str, optional
        场景描述，如 "single_node" 或 "multi_node"，默认为 "single_node"。

    model_name : str, optional
        模型名称，用于结果展示，默认为空。

    deploy_desc : str, optional
        部署描述，用于结果展示，默认为空。

    返回
    -------
    List[Dict[str, Any]]
        每个并发级别对应的性能指标字典:
        - scenario: 场景描述
        - model_name: 模型名称
        - deploy_desc: 部署描述
        - concurrency: 并发级别
        - tok_per_sec_single: 单进程吞吐
        - tok_per_sec_total: 总吞吐
        - ttft_p50_ms: 首包延迟中位数 (ms)
        - ttft_p99_ms: 首包延迟 P99 (ms)

    异常
    ------
    ValueError
        - gpu_spec 缺少必要字段
        - model_params_b <= 0
        - precision 不支持
        - concurrency_levels 为空或包含非正数
        - gpu_count <= 0
    """
    # ── 参数校验 ──────────────────────────────────────────────────────
    precision = precision.strip().lower()

    required_fields = ["memory_bw_gbs", "fp16_tflops", "memory_gb"]
    for field in required_fields:
        if field not in gpu_spec:
            raise ValueError(f"GPU 规格缺少必要字段: {field}")

    if model_params_b <= 0:
        raise ValueError(
            f"模型参数量必须为正数，当前值: {model_params_b} "
            "(model_params_b must be positive)"
        )
    if precision not in _PRECISION_BYTES:
        raise ValueError(
            f"不支持的精度 '{precision}'，可选: "
            f"{list(_PRECISION_BYTES.keys())}"
        )
    if not concurrency_levels:
        raise ValueError("并发级别列表不能为空（concurrency_levels is empty）")
    for c in concurrency_levels:
        if not isinstance(c, int) or c <= 0:
            raise ValueError(
                f"并发级别必须为正整数，当前值: {c}"
            )
    if gpu_count <= 0:
        raise ValueError("GPU 数量必须为正数")

    # ── 计算核心指标 ──────────────────────────────────────────────────
    bytes_per_param = _PRECISION_BYTES[precision]
    # 单卡模型显存占用
    model_size_per_gpu_gb = (model_params_b * bytes_per_param) / gpu_count
    # 内存带宽受限的单卡解码吞吐 (tok/s)
    memory_bw_per_gpu = gpu_spec["memory_bw_gbs"] * memory_bw_factor
    max_tok_s_per_gpu = memory_bw_per_gpu / model_size_per_gpu_gb
    # 应用折损
    max_tok_s_per_gpu_actual = max_tok_s_per_gpu * _DEGRADATION

    # ── TTFT 计算 (Prefill 阶段) ──────────────────────────────────────
    # Prefill 是计算绑定: params * 2 FLOPs / param
    total_flops = model_params_b * 2 * 1e12
    total_tflops = gpu_spec["fp16_tflops"] * gpu_count
    utilization = 0.6
    ttft_base_sec = total_flops / (total_tflops * 1e12 * utilization)
    # 多节点网络延迟 overhead (约 5-10%)
    network_overhead = 1.0 + (rdma_latency_us / 1e6) * 0.01 if gpu_count > 8 else 1.05
    ttft_p50_ms = (ttft_base_sec * network_overhead) * 1000
    ttft_p99_ms = ttft_p50_ms * 2.0

    # ── 逐并发级别计算 ────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    for concurrency in sorted(concurrency_levels):
        # 只要有一定并发就可以达到接近饱和的吞吐
        # 当并发很少时, 吞吐稍低, 但快速上升
        if concurrency < 4:
            scaling_factor = concurrency / 4.0
        else:
            scaling_factor = 1.0
        # 微小的批处理效率提升在高并发时
        batching_efficiency = 1.0 + (min(concurrency, 256) / 256) * 0.05

        tok_per_sec_total = max_tok_s_per_gpu_actual * gpu_count * scaling_factor * batching_efficiency
        tok_per_sec_single = tok_per_sec_total / concurrency if concurrency > 0 else 0.0

        results.append({
            "scenario": scenario,
            "model_name": model_name,
            "deploy_desc": deploy_desc,
            "concurrency": concurrency,
            "tok_per_sec_single": round(tok_per_sec_single, 2),
            "tok_per_sec_total": round(tok_per_sec_total, 2),
            "ttft_p50_ms": round(ttft_p50_ms, 2),
            "ttft_p99_ms": round(ttft_p99_ms, 2),
        })

    return results
