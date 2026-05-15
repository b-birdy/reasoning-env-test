"""性能预估单元测试。"""
import sys
import os

import pytest

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reasoning_env_test.performance.estimator import estimate_performance


class TestEstimatePerformance:
    """estimate_performance 函数单元测试。"""

    def test_single_a100_fp16_7b(self):
        """单张 A100 80GB, 7B FP16 场景：验证计算公式。"""
        hardware = [{
            "type": "nvidia",
            "model": "NVIDIA A100-SXM4-80GB",
            "memory_total_gb": 80.0,
            "compute_units": 1,
            "details": {"gpu_count": 1},
        }]
        results = estimate_performance(hardware, 7.0, "fp16", [4, 8])

        assert len(results) == 2

        # concurrency=4
        r = results[0]
        assert r["concurrency"] == 4
        # tok/s = 800 * (7/7) * 1.0 * 0.6 = 480.0
        assert r["tok_per_sec"] == pytest.approx(480.0, rel=0.01)
        # p50 = 200 * (7/7) * (4^0.3) / 0.6
        expected_p50 = 200.0 * (4 ** 0.3) / 0.6
        assert r["ttft_p50_ms"] == pytest.approx(expected_p50, rel=0.01)
        assert r["ttft_p99_ms"] == pytest.approx(expected_p50 * 2, rel=0.01)
        # max_supported = int(80 / (4 * 1.0)) = 20
        assert r["max_supported"] == 20

    def test_single_a100_int4_7b(self):
        """单张 A100 80GB, 7B INT4 场景：精度系数影响。"""
        hardware = [{
            "type": "nvidia",
            "model": "NVIDIA A100-SXM4-80GB",
            "memory_total_gb": 80.0,
            "compute_units": 1,
            "details": {"gpu_count": 1},
        }]
        results = estimate_performance(hardware, 7.0, "int4", [4])

        r = results[0]
        # tok/s = 800 * (7/7) * 1.5 * 0.6 = 720.0
        assert r["tok_per_sec"] == pytest.approx(720.0, rel=0.01)
        # max_supported = int(80 / (4 * 0.25)) = 80
        assert r["max_supported"] == 80
        # TTFT 受精度影响不大（主要看参数量和并发）
        expected_p50 = 200.0 * (4 ** 0.3) / 0.6
        assert r["ttft_p50_ms"] == pytest.approx(expected_p50, rel=0.01)

    def test_dual_a100_fp16_7b(self):
        """双张 A100 80GB, 7B FP16：验证多卡线性缩放。"""
        hardware = [
            {"type": "nvidia", "model": "NVIDIA A100-SXM4-80GB",
             "memory_total_gb": 80.0, "compute_units": 1,
             "details": {"gpu_count": 2}},
            {"type": "nvidia", "model": "NVIDIA A100-SXM4-80GB",
             "memory_total_gb": 80.0, "compute_units": 1,
             "details": {"gpu_count": 2}},
        ]
        results = estimate_performance(hardware, 7.0, "fp16", [4])

        r = results[0]
        # 双卡 tok/s 应为单卡的 2 倍: 800 * (7/7) * 1.0 * 0.6 * 2 = 960.0
        assert r["tok_per_sec"] == pytest.approx(960.0, rel=0.01)
        # 双卡显存合计 160GB, max_supported = int(160 / 4) = 40
        assert r["max_supported"] == 40

    def test_large_model_70b(self):
        """70B 模型在 A100 上, FP16 精度：验证参数量缩放与显存限制。"""
        hardware = [{
            "type": "nvidia",
            "model": "NVIDIA A100-SXM4-80GB",
            "memory_total_gb": 80.0,
            "compute_units": 1,
            "details": {"gpu_count": 1},
        }]
        results = estimate_performance(hardware, 70.0, "fp16", [4])

        r = results[0]
        # tok/s = 800 * (7/70) * 1.0 * 0.6 = 48.0
        assert r["tok_per_sec"] == pytest.approx(48.0, rel=0.01)
        # KV cache per request = 4 * (70/7) * 1.0 = 40 GB
        # max_supported = int(80 / 40) = 2
        assert r["max_supported"] == 2

    def test_all_concurrency_levels(self):
        """测试标准并发级别列表 [4,8,16,32,64,128] 的完整输出。"""
        hardware = [{
            "type": "nvidia",
            "model": "NVIDIA A100-SXM4-80GB",
            "memory_total_gb": 80.0,
            "compute_units": 1,
            "details": {"gpu_count": 1},
        }]
        levels = [4, 8, 16, 32, 64, 128]
        results = estimate_performance(hardware, 7.0, "fp16", levels)

        assert len(results) == len(levels)

        # tok/s 对所有并发级别应一致（理论模型下吞吐与并发无关）
        for r in results:
            assert r["tok_per_sec"] == pytest.approx(480.0, rel=0.01)

        # TTFT 随并发递增
        for i in range(1, len(results)):
            assert results[i]["ttft_p50_ms"] > results[i - 1]["ttft_p50_ms"]
            assert results[i]["ttft_p99_ms"] > results[i - 1]["ttft_p99_ms"]

    def test_trends_are_reasonable(self):
        """验证 INT4 下单卡各并发值趋势合理。"""
        hardware = [{
            "type": "nvidia",
            "model": "NVIDIA A100-SXM4-80GB",
            "memory_total_gb": 80.0,
            "compute_units": 1,
            "details": {"gpu_count": 1},
        }]
        results = estimate_performance(hardware, 7.0, "int4",
                                       [4, 8, 16, 32, 64, 128])

        # 所有结果的 tok/s 一致
        assert all(r["tok_per_sec"] == pytest.approx(720.0, rel=0.01)
                   for r in results)
        # 所有结果的 max_supported 一致
        assert all(r["max_supported"] == 80 for r in results)
        # TTFT 严格递增
        p50_vals = [r["ttft_p50_ms"] for r in results]
        p99_vals = [r["ttft_p99_ms"] for r in results]
        for i in range(1, len(p50_vals)):
            assert p50_vals[i] > p50_vals[i - 1], (
                f"p50 应在并发增加时递增: {p50_vals}"
            )
            assert p99_vals[i] > p99_vals[i - 1], (
                f"p99 应在并发增加时递增: {p99_vals}"
            )

    def test_int8_precision(self):
        """INT8 精度系数验证。"""
        hardware = [{
            "type": "nvidia", "model": "A100",
            "memory_total_gb": 80.0, "compute_units": 1,
            "details": {},
        }]
        results = estimate_performance(hardware, 7.0, "int8", [4])
        # tok/s = 800 * (7/7) * 1.2 * 0.6 = 576.0
        assert results[0]["tok_per_sec"] == pytest.approx(576.0, rel=0.01)

    # ---- 异常分支 ----

    def test_invalid_precision(self):
        """不支持的精度应抛出 ValueError。"""
        hardware = [{"type": "nvidia", "model": "A100",
                     "memory_total_gb": 80.0, "compute_units": 1,
                     "details": {}}]
        with pytest.raises(ValueError, match="精度|precision"):
            estimate_performance(hardware, 7.0, "fp32", [4])

    def test_empty_hardware(self):
        """空硬件列表应抛出 ValueError。"""
        with pytest.raises(ValueError, match="硬件|hardware"):
            estimate_performance([], 7.0, "fp16", [4])

    def test_invalid_model_params_zero(self):
        """参数量为 0 应抛出 ValueError。"""
        hardware = [{"type": "nvidia", "model": "A100",
                     "memory_total_gb": 80.0, "compute_units": 1,
                     "details": {}}]
        with pytest.raises(ValueError, match="参数量|params"):
            estimate_performance(hardware, 0.0, "fp16", [4])

    def test_invalid_model_params_negative(self):
        """参数量为负数应抛出 ValueError。"""
        hardware = [{"type": "nvidia", "model": "A100",
                     "memory_total_gb": 80.0, "compute_units": 1,
                     "details": {}}]
        with pytest.raises(ValueError, match="参数量|params"):
            estimate_performance(hardware, -1.0, "fp16", [4])

    def test_empty_concurrency_levels(self):
        """空并发列表应抛出 ValueError。"""
        hardware = [{"type": "nvidia", "model": "A100",
                     "memory_total_gb": 80.0, "compute_units": 1,
                     "details": {}}]
        with pytest.raises(ValueError, match="并发|concurrency"):
            estimate_performance(hardware, 7.0, "fp16", [])

    def test_negative_concurrency(self):
        """负并发值应抛出 ValueError。"""
        hardware = [{"type": "nvidia", "model": "A100",
                     "memory_total_gb": 80.0, "compute_units": 1,
                     "details": {}}]
        with pytest.raises(ValueError, match="并发|concurrency"):
            estimate_performance(hardware, 7.0, "fp16", [-1, 4])

    def test_output_format(self):
        """验证输出字典包含所有期望字段。"""
        hardware = [{
            "type": "nvidia", "model": "A100",
            "memory_total_gb": 80.0, "compute_units": 1,
            "details": {},
        }]
        results = estimate_performance(hardware, 7.0, "fp16", [4])
        r = results[0]
        expected_keys = {"concurrency", "tok_per_sec", "ttft_p50_ms",
                         "ttft_p99_ms", "max_supported"}
        assert expected_keys.issubset(r.keys())
        assert isinstance(r["concurrency"], int)
        assert isinstance(r["tok_per_sec"], float)
        assert isinstance(r["ttft_p50_ms"], float)
        assert isinstance(r["ttft_p99_ms"], float)
        assert isinstance(r["max_supported"], int)
