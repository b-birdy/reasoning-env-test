"""模型推荐器单元测试。"""
import sys
import os
from unittest.mock import patch, mock_open, MagicMock

import pytest

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reasoning_env_test.recommend.recommender import ModelRecommender


# ============================================================
# 模型推荐器测试
# ============================================================

class TestModelRecommender:
    def test_init_24gb_single_gpu(self):
        """测试单 24GB 显卡场景：匹配7B/13B FP16, 70B INT4。"""
        hardware_results = [
            {"type": "cpu", "model": "CPU", "memory_total_gb": 16, "compute_units": 8, "details": {}},
            {"type": "nvidia", "model": "RTX 3090", "memory_total_gb": 24, "compute_units": 10496, "details": {"gpu_count": 1}},
        ]

        with patch('builtins.open', mock_open(read_data='[{"name": "Qwen2.5-7B-Instruct", "params_b": 7.0, "min_vram_fp16_gb": 16.0, "min_vram_int8_gb": 8.0, "min_vram_int4_gb": 5.0}, {"name": "Qwen2.5-14B-Instruct", "params_b": 14.0, "min_vram_fp16_gb": 31.0, "min_vram_int8_gb": 16.0, "min_vram_int4_gb": 9.0}, {"name": "Llama-3.1-70B-Instruct", "params_b": 70.0, "min_vram_fp16_gb": 155.0, "min_vram_int8_gb": 78.0, "min_vram_int4_gb": 43.0}]')):
            recommender = ModelRecommender(hardware_results)
            result = recommender.recommend()
            assert "error" not in result or not result["error"]

    def test_4gb_no_match(self):
        """测试 4GB 显卡场景：无匹配，给出提示。"""
        hardware_results = [
            {"type": "cpu", "model": "CPU", "memory_total_gb": 16, "compute_units": 8, "details": {}},
            {"type": "nvidia", "model": "RTX 3050", "memory_total_gb": 4.0, "compute_units": 2560, "details": {"gpu_count": 1}},
        ]

        with patch('builtins.open', mock_open(read_data='[{"name": "Qwen2.5-72B-Instruct", "params_b": 72.0, "min_vram_fp16_gb": 159.0, "min_vram_int8_gb": 80.0, "min_vram_int4_gb": 44.0}]')):
            recommender = ModelRecommender(hardware_results)
            result = recommender.recommend()
            assert "error" in result and result["error"]
            assert "min_required" in result

    def test_multi_gpu_48gb(self):
        """测试多 GPU 场景：2×24GB=48GB，匹配更大模型。"""
        hardware_results = [
            {"type": "cpu", "model": "CPU", "memory_total_gb": 16, "compute_units": 8, "details": {}},
            {"type": "nvidia", "model": "RTX 3090", "memory_total_gb": 48.0, "compute_units": 20992, "details": {"gpu_count": 2}},
        ]

        with patch('builtins.open', mock_open(read_data='[{"name": "Qwen2.5-32B-Instruct", "params_b": 32.0, "min_vram_fp16_gb": 71.0, "min_vram_int8_gb": 36.0, "min_vram_int4_gb": 20.0}, {"name": "Qwen2.5-14B-Instruct", "params_b": 14.0, "min_vram_fp16_gb": 31.0, "min_vram_int8_gb": 16.0, "min_vram_int4_gb": 9.0}]')):
            recommender = ModelRecommender(hardware_results)
            result = recommender.recommend()
            assert "error" not in result or not result["error"]

    def test_mixed_gpu_nvidia_amd(self):
        """测试混合 GPU 场景：NVIDIA+AMD 分别推荐。"""
        hardware_results = [
            {"type": "cpu", "model": "CPU", "memory_total_gb": 16, "compute_units": 8, "details": {}},
            {"type": "nvidia", "model": "RTX 3090", "memory_total_gb": 24, "compute_units": 10496, "details": {"gpu_count": 1}},
            {"type": "amd", "model": "MI250X", "memory_total_gb": 64, "compute_units": 220, "details": {"gpu_count": 1}},
        ]

        with patch('builtins.open', mock_open(read_data='[{"name": "Qwen2.5-7B-Instruct", "params_b": 7.0, "min_vram_fp16_gb": 16.0, "min_vram_int8_gb": 8.0, "min_vram_int4_gb": 5.0}]')):
            recommender = ModelRecommender(hardware_results)
            result = recommender.recommend()
            assert "error" not in result or not result["error"]
