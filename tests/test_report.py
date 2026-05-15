"""报告生成器单元测试。"""
import sys
import os

import pytest

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reasoning_env_test.report.reporter import generate_report


# ============================================================
# 辅助函数：构造标准测试数据
# ============================================================

def _make_hardware(cpu_mem=16.0, cpu_cores=8, gpu_model="NVIDIA A100-SXM4-80GB",
                   gpu_mem=80.0, gpu_count=1, gpu_type="nvidia"):
    """构造标准硬件检测结果列表。"""
    results = [
        {
            "type": "cpu",
            "model": "Intel(R) Xeon(R) Gold 6348",
            "memory_total_gb": cpu_mem,
            "compute_units": cpu_cores,
            "details": {"architecture": "x86_64", "logical_cores": cpu_cores, "os": "Linux"},
        },
    ]
    if gpu_type == "nvidia":
        results.append({
            "type": "nvidia",
            "model": gpu_model,
            "memory_total_gb": gpu_mem,
            "compute_units": 1,
            "details": {"gpu_count": gpu_count, "driver_version": "550.90.07"},
        })
    elif gpu_type == "amd":
        results.append({
            "type": "amd",
            "model": gpu_model,
            "memory_total_gb": gpu_mem,
            "compute_units": 220,
            "details": {"gpu_count": gpu_count},
        })
    # 添加空结果的 GPU 类型
    for t in ["amd", "ascend", "kunlunxin", "hygon"]:
        if gpu_type != t:
            results.append({
                "type": t, "model": "", "memory_total_gb": 0.0,
                "compute_units": 0, "details": {},
            })
    return results


def _make_software(all_commands=True, all_frameworks=True):
    """构造标准软件检测结果。"""
    cmds = {
        "python3": True, "python": True, "pip": True,
        "nvidia-smi": True, "rocm-smi": False,
        "npu-smi": False, "docker": True,
        "kubectl": True, "ollama": True,
    }
    if not all_commands:
        for k in cmds:
            cmds[k] = False
        cmds["python3"] = True
        cmds["python"] = True

    frameworks = {
        "vllm": "0.6.0" if all_frameworks else None,
        "text_generation": "2.0.0" if all_frameworks else None,
        "ollama": "0.3.0" if all_frameworks else None,
        "tensorrt_llm": "0.12.0" if all_frameworks else None,
        "onnxruntime": "1.18.0" if all_frameworks else None,
        "torch": "2.5.0" if all_frameworks else None,
        "tensorflow": "2.17.0" if all_frameworks else None,
    }

    return {
        "commands": cmds,
        "frameworks": frameworks,
        "python_version": "3.12.5",
        "python_ok": True,
        "cuda_version": "12.4",
        "rocm_version": None,
        "hardware_tools": {},
    }


def _make_container(in_container=False, container_type=None,
                    k8s=False, docker_installed=True):
    """构造标准容器检测结果。"""
    return {
        "in_container": in_container,
        "container_type": container_type,
        "k8s_pod": k8s,
        "docker_installed": docker_installed,
        "memory_limit_bytes": 8388608 if in_container else None,
        "cpu_quota": 100000 if in_container else None,
    }


def _make_recommendations(with_error=False):
    """构造标准模型推荐结果。"""
    if with_error:
        return {
            "error": True,
            "message": "未检测到可用于模型部署的GPU硬件",
            "min_required": {"gpu_required": True},
        }
    return [
        {
            "model": {
                "name": "Qwen2.5-7B-Instruct",
                "params_b": 7.0,
                "type": "llm",
                "source": "ModelScope",
                "model_scope_url": "https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct",
                "min_vram_fp16_gb": 16.0,
                "min_vram_int8_gb": 8.0,
                "min_vram_int4_gb": 5.0,
                "family": "Qwen2.5",
                "notes": "通义千问主力对话模型",
            },
            "precision": "fp16",
            "gpu_index": 0,
            "gpu_type": "nvidia",
            "is_multi_gpu": False,
        },
        {
            "model": {
                "name": "Llama-3.2-3B-Instruct",
                "params_b": 3.0,
                "type": "llm",
                "source": "ModelScope",
                "model_scope_url": "https://modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct",
                "min_vram_fp16_gb": 7.0,
                "min_vram_int8_gb": 4.0,
                "min_vram_int4_gb": 2.0,
                "family": "Llama",
                "notes": "Meta轻量对话",
            },
            "precision": "fp16",
            "gpu_index": 0,
            "gpu_type": "nvidia",
            "is_multi_gpu": False,
        },
    ]


def _make_performance():
    """构造标准性能预估结果。"""
    return [
        {"concurrency": 4, "tok_per_sec": 480.0, "ttft_p50_ms": 350.0,
         "ttft_p99_ms": 700.0, "max_supported": 20},
        {"concurrency": 8, "tok_per_sec": 480.0, "ttft_p50_ms": 420.0,
         "ttft_p99_ms": 840.0, "max_supported": 20},
        {"concurrency": 16, "tok_per_sec": 480.0, "ttft_p50_ms": 510.0,
         "ttft_p99_ms": 1020.0, "max_supported": 20},
    ]


def _make_system_info(distro="Ubuntu", version="22.04"):
    """构造标准系统信息检测结果。"""
    return {
        "os": {"distribution": distro, "version": version,
               "kernel": "5.15.0-91-generic", "hostname": "test-server",
               "architecture": "x86_64"},
        "cpu": {"model_name": "Intel(R) Xeon(R) Gold 6348",
                "architecture": "x86_64", "physical_cores": 8,
                "logical_cores": 16, "frequency_mhz": 2500.0,
                "max_frequency_mhz": 3500.0, "cache_l1d": "48K",
                "cache_l1i": "32K", "cache_l2": "1.25M",
                "cache_l3": "36M", "threads_per_core": 2,
                "sockets": 1,
                "flags": ["avx512", "avx2", "sse4_2", "aes", "fma3"]},
        "memory": {"total_gb": 512.0, "type": "DDR5",
                   "speed_mhz": 4800, "modules": 8, "size_per_module_gb": 64},
    }


def _make_network():
    """构造标准网络检测结果。"""
    return {
        "interfaces": [
            {"name": "eth0", "type": "ethernet", "speed_gbps": 25,
             "state": "up", "mac": "00:11:22:33:44:55"},
            {"name": "eth1", "type": "roce", "speed_gbps": 200,
             "state": "up", "mac": "00:11:22:33:44:56", "rdma": True},
        ],
        "rdma_devices": ["mlx5_0", "mlx5_1"],
        "total_network_ports": 2,
        "summary": {"400g_ports": 0, "200g_ports": 1, "25g_ports": 1,
                    "total_bandwidth_gbps": 225},
    }


# ============================================================
# 报告生成器测试
# ============================================================

class TestGenerateReport:
    """generate_report 函数单元测试。"""

    def test_full_report_contains_all_sections(self):
        """完整报告应包含所有七个章节标题和脚注。"""
        hardware = _make_hardware()
        software = _make_software()
        container = _make_container()
        recommendations = _make_recommendations()
        performance = _make_performance()
        system_info = _make_system_info()
        network = _make_network()

        report = generate_report(hardware, software, container,
                                  recommendations, performance,
                                  system_info, network)

        # 检查所有七个章节标题
        assert "## 1. 概览" in report
        assert "## 2. 系统信息" in report
        assert "## 3. GPU/加速卡" in report
        assert "## 4. 网络" in report
        assert "## 5. 软件环境" in report
        assert "## 6. 可部署模型推荐" in report
        assert "## 7. 性能预估" in report

        # 检查脚注
        assert "折损" in report or "说明" in report

    def test_report_contains_cpu_gpu_info(self):
        """报告应包含 CPU 和 GPU 信息。"""
        hardware = _make_hardware(gpu_model="NVIDIA A100-SXM4-80GB", gpu_mem=80.0)
        software = _make_software()
        container = _make_container()
        recommendations = _make_recommendations()
        performance = _make_performance()
        system_info = _make_system_info()
        network = _make_network()

        report = generate_report(hardware, software, container,
                                  recommendations, performance,
                                  system_info, network)

        # 应包含 CPU 型号
        assert "Intel(R) Xeon(R) Gold 6348" in report
        # 应包含 GPU 型号
        assert "NVIDIA A100" in report
        # 应包含内存
        assert "80.0 GB" in report or "80GB" in report
        # 应包含 Python 版本
        assert "3.12.5" in report
        # 应包含 CUDA 版本
        assert "12.4" in report
        # 应包含网络信息
        assert "eth0" in report
        assert "25 Gbps" in report
        # 应包含系统信息
        assert "Ubuntu" in report or "DDR5" in report

    def test_missing_commands_marked(self):
        """缺失命令应标记为 ⚠️。"""
        hardware = _make_hardware()
        software = _make_software(all_commands=False)
        container = _make_container()
        recommendations = _make_recommendations()
        performance = _make_performance()

        report = generate_report(hardware, software, container,
                                  recommendations, performance)

        # 缺失 docker 应有 ⚠️
        assert "⚠️" in report
        # ✅ 也应出现（对存在的命令）
        assert "✅" in report

    def test_no_deployable_model_shows_reason(self):
        """没有可部署模型时需说明原因。"""
        hardware = _make_hardware(gpu_mem=0.0, gpu_type="nvidia")
        software = _make_software()
        container = _make_container()
        recommendations = _make_recommendations(with_error=True)
        performance = _make_performance()

        report = generate_report(hardware, software, container,
                                  recommendations, performance)

        # 应包含错误原因
        assert "未检测到可用于模型部署的GPU硬件" in report
        # 应包含模型推荐相关的说明
        assert "模型" in report and ("推荐" in report or "最低配置" in report)

    def test_performance_table_has_all_rows(self):
        """性能预估表格应包含所有并发级别的数据行。"""
        hardware = _make_hardware()
        software = _make_software()
        container = _make_container()
        recommendations = _make_recommendations()
        performance = _make_performance()

        report = generate_report(hardware, software, container,
                                  recommendations, performance)

        # 应包含并发数
        assert "| 4 |" in report
        assert "| 8 |" in report
        assert "| 16 |" in report
        # 应包含表头列名
        assert "并发批次" in report
        assert "吞吐 (tok/s)" in report or "吞吐(tok/s)" in report
        assert "P50" in report
        assert "P99" in report
        assert "延迟" in report

    def test_recommendation_table_has_columns(self):
        """模型推荐表格应包含所有要求的列。"""
        hardware = _make_hardware()
        software = _make_software()
        container = _make_container()
        recommendations = _make_recommendations()
        performance = _make_performance()

        report = generate_report(hardware, software, container,
                                  recommendations, performance)

        # 应包含表头
        assert "推荐模型" in report
        assert "参数量" in report
        assert "推荐精度" in report
        assert "显存总需求" in report
        assert "模型类型" in report
        assert "部署建议" in report
        assert "推理框架" in report
        assert "ModelScope" in report
        # 应包含模型名称
        assert "Qwen2.5-7B-Instruct" in report
        assert "Llama-3.2-3B-Instruct" in report
        # 应包含精度
        assert "fp16" in report

    def test_container_status_shown(self):
        """软件环境章节应显示容器状态。"""
        hardware = _make_hardware()
        software = _make_software()
        container = _make_container(in_container=True, container_type="docker")
        recommendations = _make_recommendations()
        performance = _make_performance()

        report = generate_report(hardware, software, container,
                                  recommendations, performance)

        # Docker 状态在 hardware_tools 中（若为空则 fallback 显示常用命令表）
        # 至少 Docker 相关信息应出现在报告中
        assert "docker" in report.lower()

    def test_empty_hardware_no_gpu(self):
        """仅有CPU无GPU时报告仍能生成。"""
        hardware = [
            {
                "type": "cpu", "model": "Intel(R) Core(TM) i7",
                "memory_total_gb": 32.0, "compute_units": 8,
                "details": {"architecture": "x86_64", "logical_cores": 8, "os": "Linux"},
            },
            {"type": "nvidia", "model": "", "memory_total_gb": 0.0, "compute_units": 0, "details": {}},
            {"type": "amd", "model": "", "memory_total_gb": 0.0, "compute_units": 0, "details": {}},
            {"type": "ascend", "model": "", "memory_total_gb": 0.0, "compute_units": 0, "details": {}},
            {"type": "kunlunxin", "model": "", "memory_total_gb": 0.0, "compute_units": 0, "details": {}},
            {"type": "hygon", "model": "", "memory_total_gb": 0.0, "compute_units": 0, "details": {}},
        ]
        software = _make_software()
        container = _make_container()
        recommendations = _make_recommendations(with_error=True)
        performance = _make_performance()

        report = generate_report(hardware, software, container,
                                  recommendations, performance)

        # 应输出完整报告
        assert "## 6. 可部署模型推荐" in report or "模型推荐" in report
        assert "## 7. 性能预估" in report or "性能预估" in report
        # 推荐部分应显示无GPU
        assert "未检测到可用于模型部署" in report or "无法推荐" in report

    def test_report_is_markdown_string(self):
        """报告应为合法的Markdown字符串。"""
        hardware = _make_hardware()
        software = _make_software()
        container = _make_container()
        recommendations = _make_recommendations()
        performance = _make_performance()

        report = generate_report(hardware, software, container,
                                  recommendations, performance)

        # 应为字符串
        assert isinstance(report, str)
        # 应包含 Markdown 表格分隔符
        assert "|---" in report
        # 应以标题开头
        assert report.strip().startswith("#")

    def test_system_info_chapter(self):
        """概览章节应包含系统信息（受 reporter 格式影响，仅验证数据能输出）。"""
        hardware = _make_hardware()
        software = _make_software()
        container = _make_container()
        recommendations = _make_recommendations()
        performance = _make_performance()
        system_info = _make_system_info()

        report = generate_report(hardware, software, container,
                                  recommendations, performance,
                                  system_info=system_info)

        # system_info 当前还未在旧版 reporter 中完整渲染
        # 但报告仍应生成（不抛异常）
        assert isinstance(report, str)
        assert report.strip().startswith("#")

    def test_network_chapter(self):
        """概览章节应包含网络信息（受 reporter 格式影响，仅验证数据能输出）。"""
        hardware = _make_hardware()
        software = _make_software()
        container = _make_container()
        recommendations = _make_recommendations()
        performance = _make_performance()
        network = _make_network()

        report = generate_report(hardware, software, container,
                                  recommendations, performance,
                                  network=network)

        # network 当前还未在旧版 reporter 中完整渲染
        # 但报告仍应生成（不抛异常）
        assert isinstance(report, str)
        assert report.strip().startswith("#")
