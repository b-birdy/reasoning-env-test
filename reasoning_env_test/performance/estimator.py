"""性能预估模块。

基于硬件信息（统一格式字典列表），估算指定模型在多并发下的
吞吐量（tok/s）和首包延迟（TTFT）。

⚠️ 重要提示
    所有数值均为 **理论估算值**，基于 40% 理论到实际性能折损。
    参考基准为 NVIDIA A100 80GB 在 FP16 精度下对 7B 模型的
    ~800 tok/s 实测数据。
    **实际性能** 可能因硬件型号、驱动版本、软件栈、工作负载特征、
    批处理策略等而与估算值存在显著差异。
    本模块的输出 **不可替代** 真实环境下的性能基准测试。
"""
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 基准参考值
# ---------------------------------------------------------------------------
# FP16 基准: 单 GPU (NVIDIA A100 80GB) 对 7B 模型的 tok/s
_BASE_TOK_PER_SEC: float = 800.0
_BASE_PARAMS_B: float = 7.0
# 每个请求的 KV cache 显存占用（7B FP16）
_BASE_KV_CACHE_GB: float = 4.0

# ---------------------------------------------------------------------------
# 精度系数
# ---------------------------------------------------------------------------
# tok/s 缩放系数 —— 低位宽精度可提升计算吞吐
_PRECISION_TOK_FACTOR: Dict[str, float] = {
    "fp16": 1.0,
    "int8": 1.2,
    "int4": 1.5,
}

# KV cache 显存占用缩放系数 —— 低位宽精度减少 KV cache 占用
_PRECISION_KV_FACTOR: Dict[str, float] = {
    "fp16": 1.0,
    "int8": 0.5,
    "int4": 0.25,
}

# ---------------------------------------------------------------------------
# 理论到实际折损
# ---------------------------------------------------------------------------
# 40% 折损：实际吞吐 = 理论 × 0.6
_DEGRADATION: float = 0.6
# 延迟折损：实际延迟 = 理论 / 0.6 （延迟变高）
_LATENCY_DEGRADATION: float = 1.0 / _DEGRADATION


def estimate_performance(
    hardware_info: List[Dict[str, Any]],
    model_params_b: float,
    precision: str,
    concurrency_levels: List[int],
) -> List[Dict[str, Any]]:
    """估算多并发下性能指标。

    参数
    ----------
    hardware_info : List[Dict[str, Any]]
        硬件统一格式字典列表。每个字典包含:
        - type: 硬件类型（如 "nvidia", "amd", "cpu"）
        - model: 型号名称
        - memory_total_gb: 总显存/内存 (GB)
        - compute_units: 计算单元数（GPU 个数 / CPU 核心数）
        - details: 详细信息字典

    model_params_b : float
        模型参数量，以十亿（B）为单位。如 7B 则传入 7.0。

    precision : str
        推理精度。可选: "fp16", "int8", "int4"。

    concurrency_levels : List[int]
        需要估算的并发级别列表，如 [4, 8, 16, 32, 64, 128]。

    返回
    -------
    List[Dict[str, Any]]
        每个并发级别对应的性能指标字典，按 concurrency 升序排列:
        - concurrency: 并发级别
        - tok_per_sec: 系统总吞吐 (tokens/s)
        - ttft_p50_ms: 首包延迟中位数 (ms)
        - ttft_p99_ms: 首包延迟 P99 (ms)
        - max_supported: 当前硬件可支持的最大并发数（由显存限制）

    异常
    ------
    ValueError
        - hardware_info 为空
        - model_params_b <= 0
        - precision 不支持
        - concurrency_levels 为空或包含非正数

    参考公式
    ---------
    tok/s = 基准_tok/s × (基准参数量 / 当前参数量) × 精度系数 × 0.6
    TTFT_p50 = 200ms × (当前参数量 / 7B) × 并发^0.3 / 0.6
    TTFT_p99 = TTFT_p50 × 2
    最大并发 = Σ(显存) / (每个请求 KV cache 占用)
    """
    # ── 参数校验 ──────────────────────────────────────────────────────
    precision = precision.strip().lower()

    if not hardware_info:
        raise ValueError("硬件信息列表不能为空（hardware_info is empty）")
    if model_params_b <= 0:
        raise ValueError(
            f"模型参数量必须为正数，当前值: {model_params_b} "
            "(model_params_b must be positive)"
        )
    if precision not in _PRECISION_TOK_FACTOR:
        raise ValueError(
            f"不支持的精度 '{precision}'，可选: "
            f"{list(_PRECISION_TOK_FACTOR.keys())}"
        )
    if not concurrency_levels:
        raise ValueError("并发级别列表不能为空（concurrency_levels is empty）")
    for c in concurrency_levels:
        if not isinstance(c, int) or c <= 0:
            raise ValueError(
                f"并发级别必须为正整数，当前值: {c}"
            )

    # ── 汇总硬件资源 ──────────────────────────────────────────────────
    total_compute_units: int = sum(
        hw.get("compute_units", 0) for hw in hardware_info
    )
    total_memory_gb: float = sum(
        hw.get("memory_total_gb", 0.0) for hw in hardware_info
    )

    # ── 精度系数 ──────────────────────────────────────────────────────
    tok_factor: float = _PRECISION_TOK_FACTOR[precision]
    kv_factor: float = _PRECISION_KV_FACTOR[precision]

    # ── 单计算单元理论 tok/s ────────────────────────────────────────
    # 公式: 基准 × (基准参数量 / 当前参数量) × 精度系数
    tok_per_unit = (
        _BASE_TOK_PER_SEC
        * (_BASE_PARAMS_B / model_params_b)
        * tok_factor
    )

    # 应用 40% 折损 -> 实际 tok/s 每单元
    tok_per_unit_actual = tok_per_unit * _DEGRADATION

    # 多计算单元线性缩放 -> 系统总吞吐
    system_tok_per_sec = tok_per_unit_actual * total_compute_units

    # ── 每个请求 KV cache 显存占用 ──────────────────────────────────
    kv_cache_per_request_gb = (
        _BASE_KV_CACHE_GB
        * (model_params_b / _BASE_PARAMS_B)
        * kv_factor
    )

    # ── 最大并发（由显存限制）───────────────────────────────────────
    if kv_cache_per_request_gb > 0 and total_memory_gb > 0:
        max_supported: int = int(total_memory_gb / kv_cache_per_request_gb)
    else:
        max_supported = 0

    # ── 逐并发级别计算 ────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    for concurrency in sorted(concurrency_levels):
        # 吞吐量：简化理论模型下与并发无关（假设批处理可充分合并）
        tok_per_sec = system_tok_per_sec

        # 首包延迟 TTFT
        # p50 ≈ 200ms × (当前参数量 / 7B) × 并发^0.3 / 0.6
        ttft_base = 200.0 * (model_params_b / _BASE_PARAMS_B) * (
            concurrency ** 0.3
        )
        ttft_p50_ms = ttft_base * _LATENCY_DEGRADATION
        ttft_p99_ms = ttft_p50_ms * 2.0

        results.append({
            "concurrency": concurrency,
            "tok_per_sec": round(tok_per_sec, 2),
            "ttft_p50_ms": round(ttft_p50_ms, 2),
            "ttft_p99_ms": round(ttft_p99_ms, 2),
            "max_supported": max_supported,
        })

    return results
