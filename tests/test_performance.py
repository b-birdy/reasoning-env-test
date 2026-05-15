"""性能预估单元测试。"""
import sys
import os

import pytest

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reasoning_env_test.performance.estimator import estimate_performance, load_gpu_specs


class TestEstimatePerformance:
    """estimate_performance 函数单元测试。"""

    def test_load_gpu_specs(self):
        """测试 GPU 规格文件加载。"""
        specs = load_gpu_specs()
        assert "kunlunxin_p800" in specs
        assert "nvidia_a100_80g" in specs
        assert specs["kunlunxin_p800"]["memory_gb"] == 96

    def test_qwen27b_single_p800_fp16(self):
        """Qwen3.6-27B (27B FP16) 单卡 P800。"""
        specs = load_gpu_specs()
        p800 = specs["kunlunxin_p800"]
        results = estimate_performance(
            p800,
            model_params_b=27.0,
            precision="fp16",
            concurrency_levels=[4, 8],
            gpu_count=1,
            scenario="single_node",
            model_name="Qwen3.6-27B",
            deploy_desc="单机单卡"
        )

        assert len(results) == 2

        # 验证大致数值范围
        r = results[0]
        assert r["concurrency"] == 4
        # 模型大小 27*2=54GB, 内存带宽 3200 GB/s → 3200 / 54 ≈ 59 tok/s, 折损后 ≈ 35
        assert r["tok_per_sec_total"] > 20
        assert r["tok_per_sec_total"] < 50
        assert r["ttft_p50_ms"] > 100
        assert r["ttft_p50_ms"] < 10000

    def test_glm51_16p800_int4(self):
        """GLM-5.1 (754B INT4) 双机 16 卡 P800。"""
        specs = load_gpu_specs()
        p800 = specs["kunlunxin_p800"]
        results = estimate_performance(
            p800,
            model_params_b=754.0,
            precision="int4",
            concurrency_levels=[64, 256],
            gpu_count=16,
            memory_bw_factor=0.95,
            scenario="multi_node",
            model_name="GLM-5.1",
            deploy_desc="双机 16 卡 RDMA 分布式推理"
        )

        assert len(results) == 2
        r = results[0]
        # 预期总吞吐在 1000-1500 tok/s 左右
        assert r["tok_per_sec_total"] > 800
        assert r["tok_per_sec_total"] < 1600

    def test_deepseek685b_8p800_int8(self):
        """DeepSeek-V3.2 (685B INT8) 单机 8 卡 P800。"""
        specs = load_gpu_specs()
        p800 = specs["kunlunxin_p800"]
        results = estimate_performance(
            p800,
            model_params_b=685.0,
            precision="int8",
            concurrency_levels=[32],
            gpu_count=8,
            scenario="single_node",
            model_name="DeepSeek-V3.2",
            deploy_desc="单机 8 卡"
        )

        r = results[0]
        # 预期总吞吐在 150-300 tok/s 左右
        assert r["tok_per_sec_total"] > 100
        assert r["tok_per_sec_total"] < 350

    def test_invalid_precision(self):
        """不支持的精度应抛出 ValueError。"""
        specs = load_gpu_specs()
        with pytest.raises(ValueError, match="精度|precision"):
            estimate_performance(specs["nvidia_a100_80g"], 7.0, "fp32", [4], 1)

    def test_invalid_model_params_zero(self):
        """参数量为 0 应抛出 ValueError。"""
        specs = load_gpu_specs()
        with pytest.raises(ValueError, match="参数量|params"):
            estimate_performance(specs["nvidia_a100_80g"], 0.0, "fp16", [4], 1)

    def test_invalid_gpu_count(self):
        """GPU 数量为 0 应抛出 ValueError。"""
        specs = load_gpu_specs()
        with pytest.raises(ValueError):
            estimate_performance(specs["nvidia_a100_80g"], 7.0, "fp16", [4], 0)

    def test_output_format(self):
        """验证输出字典包含所有期望字段。"""
        specs = load_gpu_specs()
        results = estimate_performance(
            specs["nvidia_a100_80g"],
            7.0,
            "fp16",
            [4],
            1
        )
        r = results[0]
        expected_keys = {"scenario", "model_name", "deploy_desc",
                         "concurrency", "tok_per_sec_single",
                         "tok_per_sec_total", "ttft_p50_ms", "ttft_p99_ms"}
        assert expected_keys.issubset(r.keys())
