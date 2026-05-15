"""集成测试 — 端到端覆盖完整运行流程与边缘情况。

测试策略：
1. subprocess 调用：通过子进程实际执行 CLI，验证退出码、输出格式、文件写入
2. mock 注入：模拟硬件缺失、检测失败等场景，验证降级行为
3. 所有测试均不修改业务模块代码，不引入新依赖
"""
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from reasoning_env_test.__main__ import main


# ============================================================
# 辅助函数
# ============================================================

def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """通过 subprocess 运行 CLI。

    注意：Windows GBK 环境下管道编码可能不一致，使用 errors='replace'
    确保即使部分字符解码失败也能返回内容。
    """
    return subprocess.run(
        [sys.executable, "-m", "reasoning_env_test", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT_ROOT,
        timeout=60,
    )


def _make_hardware(cpu_mem=16.0):
    """CPU-only 硬件结果。"""
    return [
        {"type": "cpu", "model": "Test CPU", "memory_total_gb": cpu_mem,
         "compute_units": 8, "details": {"architecture": "x86_64"}},
    ]


def _make_hardware_with_gpu():
    """含 NVIDIA GPU 的硬件结果。"""
    return [
        {"type": "cpu", "model": "Intel(R) Xeon(R) Gold 6348",
         "memory_total_gb": 16.0, "compute_units": 8,
         "details": {"architecture": "x86_64"}},
        {"type": "nvidia", "model": "NVIDIA A100-SXM4-80GB",
         "memory_total_gb": 80.0, "compute_units": 1,
         "details": {"gpu_count": 1, "driver_version": "550.90.07"}},
        {"type": "amd", "model": "", "memory_total_gb": 0.0,
         "compute_units": 0, "details": {}},
        {"type": "ascend", "model": "", "memory_total_gb": 0.0,
         "compute_units": 0, "details": {}},
        {"type": "kunlunxin", "model": "", "memory_total_gb": 0.0,
         "compute_units": 0, "details": {}},
        {"type": "hygon", "model": "", "memory_total_gb": 0.0,
         "compute_units": 0, "details": {}},
    ]


def _make_software():
    return {
        "commands": {"python3": True, "nvidia-smi": False},
        "frameworks": {"torch": "2.0.0"},
        "python_version": "3.10.0",
        "python_ok": True,
        "cuda_version": None,
        "rocm_version": None,
    }


def _make_container():
    return {
        "in_container": False, "container_type": None,
        "k8s_pod": False, "docker_installed": True,
        "memory_limit_bytes": None, "cpu_quota": None,
    }


# ============================================================
# subprocess 集成测试（真实调用）
# ============================================================

class TestSubprocessIntegration:
    """通过 subprocess 调用 CLI 的端到端测试。

    注意：实际环境可能无 GPU，这些测试验证工具在真实机器上的行为。
    """

    def test_cli_exit_code(self):
        """CLI 能以退出码 0 或 1 正常结束（取决于环境）。"""
        result = _run_cli()
        # 无论有无 GPU 都应正常退出
        assert result.returncode in (0, 1), (
            f"returncode={result.returncode}, stderr={result.stderr[:500]}"
        )

    def test_cli_output_contains_markdown(self):
        """stdout 输出应包含 Markdown 报告基本结构。"""
        result = _run_cli()
        assert result.stdout is not None, "stdout 不应为 None"
        # 报告应包含章节标题或 Markdown 格式标记
        assert "#" in result.stdout or "|" in result.stdout, \
            "stdout 应包含 Markdown 内容"

    def test_cli_output_file(self, tmp_path):
        """-o 参数写入文件，内容应有效。"""
        output_file = tmp_path / "report.md"
        result_file = _run_cli("-o", str(output_file))

        assert result_file.returncode in (0, 1)
        assert output_file.exists(), "输出文件应被创建"
        content = output_file.read_text(encoding="utf-8")
        assert len(content) > 0, "输出文件不应为空"
        # 内容应有报告基本结构
        assert "#" in content or "|" in content

    def test_cli_verbose_output(self):
        """-v 模式应在 stderr 产生日志输出。"""
        result = _run_cli("-v")
        assert result.returncode in (0, 1)
        # verbose 模式下 stderr 应有 INFO 日志
        assert "[INFO]" in result.stderr or "INFO" in result.stderr, \
            "verbose 模式 stderr 应有日志"

    def test_cli_help(self):
        """--help 应打印帮助信息并退出 0。"""
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "--output" in result.stdout or "-o" in result.stdout, \
            "帮助信息应包含 --output 参数说明"

    def test_cli_output_file_cleans_up(self, tmp_path):
        """-o 输出到临时目录后文件内容正确。"""
        output_file = tmp_path / "report.md"
        result = _run_cli("-o", str(output_file))
        assert result.returncode in (0, 1)
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert len(content) > 0, "输出文件不应为空"


# ============================================================
# Mock 集成测试（场景模拟）
# ============================================================


class TestIntegrationFullFlow:
    """全流程 mock 集成测试。"""

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_full_flow_with_gpu(
        self, mock_container, mock_software, mock_hardware,
    ):
        """完整流程：含 NVIDIA GPU，推荐模型 + 性能预估 + 报告均正常。"""
        mock_hardware.return_value = _make_hardware_with_gpu()
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        with patch("sys.stdout") as mock_stdout:
            rc = main([])
            assert rc == 0, f"含 GPU 全流程应返回 0，实际 {rc}"
            assert mock_stdout.buffer.write.called, "应输出报告"

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_full_flow_to_file_with_gpu(
        self, mock_container, mock_software, mock_hardware, tmp_path,
    ):
        """含 GPU 全流程输出到文件。"""
        mock_hardware.return_value = _make_hardware_with_gpu()
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        output_file = tmp_path / "report.md"
        rc = main(["-o", str(output_file)])
        assert rc == 0
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        # 应包含报告的基本章节
        assert "推理环境检测报告" in content or "## " in content


class TestIntegrationNoGpu:
    """无 GPU 场景集成测试。

    CPU-only 不是错误状态——工具检测到无 GPU 时会生成报告说明
    「无可部署模型」，并以退出码 0 正常结束。
    """

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_cpu_only_still_generates_report(
        self, mock_container, mock_software, mock_hardware,
    ):
        """纯 CPU 环境：报告仍应生成，推荐部分应有说明。"""
        mock_hardware.return_value = _make_hardware()
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        with patch("sys.stdout") as mock_stdout:
            rc = main([])
            # CPU-only 无异常 → 正常退出 0
            assert rc == 0, f"CPU-only 应返回 0，实际 {rc}"
            assert mock_stdout.buffer.write.called, "CPU-only 也应输出报告"

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_cpu_only_output_file(
        self, mock_container, mock_software, mock_hardware, tmp_path,
    ):
        """纯 CPU 环境：输出到文件。"""
        mock_hardware.return_value = _make_hardware(cpu_mem=32.0)
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        output_file = tmp_path / "cpu_report.md"
        rc = main(["-o", str(output_file)])
        assert rc == 0
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert len(content) > 0

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_low_memory_cpu(
        self, mock_container, mock_software, mock_hardware,
    ):
        """小内存 CPU 环境：检测仍正常工作。"""
        mock_hardware.return_value = _make_hardware(cpu_mem=2.0)
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        with patch("sys.stdout"):
            rc = main(["-v"])
            assert rc == 0, f"低内存场景应返回 0，实际 {rc}"


class TestIntegrationPartialFailure:
    """部分模块失败场景集成测试。"""

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_hardware_failure_others_ok(
        self, mock_container, mock_software, mock_hardware,
    ):
        """硬件检测失败：软件/容器检测不受影响，报告仍生成。"""
        mock_hardware.side_effect = RuntimeError("no hardware access")
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        with patch("sys.stdout") as mock_stdout:
            rc = main([])
            assert rc == 1, f"硬件失败应返回 1，实际 {rc}"
            # 报告仍应输出（降级数据）
            assert mock_stdout.buffer.write.called

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_software_failure_others_ok(
        self, mock_container, mock_software, mock_hardware,
    ):
        """软件检测失败：硬件/容器检测不受影响。"""
        mock_hardware.return_value = _make_hardware_with_gpu()
        mock_software.side_effect = RuntimeError("software detection crashed")
        mock_container.return_value = _make_container()

        with patch("sys.stdout") as mock_stdout:
            rc = main([])
            assert rc == 1, f"软件失败应返回 1，实际 {rc}"
            assert mock_stdout.buffer.write.called

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_container_failure_others_ok(
        self, mock_container, mock_software, mock_hardware,
    ):
        """容器检测失败：硬件/软件检测不受影响。"""
        mock_hardware.return_value = _make_hardware_with_gpu()
        mock_software.return_value = _make_software()
        mock_container.side_effect = RuntimeError("container detection crashed")

        with patch("sys.stdout") as mock_stdout:
            rc = main([])
            assert rc == 1, f"容器失败应返回 1，实际 {rc}"
            assert mock_stdout.buffer.write.called

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_all_detectors_fail(
        self, mock_container, mock_software, mock_hardware,
    ):
        """所有检测模块均失败：报告仍应生成（全部降级为空数据）。"""
        mock_hardware.side_effect = RuntimeError("hardware fail")
        mock_software.side_effect = RuntimeError("software fail")
        mock_container.side_effect = RuntimeError("container fail")

        with patch("sys.stdout") as mock_stdout:
            rc = main([])
            assert rc == 1, f"全失败应返回 1，实际 {rc}"
            # 报告生成模块应仍被调用
            assert mock_stdout.buffer.write.called

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_output_file_after_partial_failure(
        self, mock_container, mock_software, mock_hardware, tmp_path,
    ):
        """部分检测失败时，输出到文件仍正常工作。"""
        mock_hardware.side_effect = RuntimeError("hardware fail")
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        output_file = tmp_path / "partial_fail_report.md"
        rc = main(["-o", str(output_file)])
        assert rc == 1
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert len(content) > 0


class TestIntegrationEdgeCases:
    """边缘情况集成测试。"""

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_empty_software_frameworks(
        self, mock_container, mock_software, mock_hardware,
    ):
        """软件检测返回空框架字典时不应崩溃（无异常=0）。"""
        mock_hardware.return_value = _make_hardware()
        mock_software.return_value = {
            "commands": {}, "frameworks": {},
            "python_version": "", "python_ok": False,
            "cuda_version": None, "rocm_version": None,
        }
        mock_container.return_value = _make_container()

        with patch("sys.stdout"):
            rc = main([])
            assert rc == 0, "空框架数据不是异常，应返回 0"

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_empty_container(
        self, mock_container, mock_software, mock_hardware,
    ):
        """容器检测返回空字典时不应崩溃（无异常=0）。"""
        mock_hardware.return_value = _make_hardware()
        mock_software.return_value = _make_software()
        mock_container.return_value = {}

        with patch("sys.stdout"):
            rc = main([])
            assert rc == 0, "空容器数据不是异常，应返回 0"

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_mixed_precision_hardware(
        self, mock_container, mock_software, mock_hardware,
    ):
        """多种 GPU 类型混合（NVIDIA + AMD）时正常工作。"""
        mock_hardware.return_value = [
            {"type": "cpu", "model": "Test CPU", "memory_total_gb": 64.0,
             "compute_units": 16, "details": {"architecture": "x86_64"}},
            {"type": "nvidia", "model": "NVIDIA RTX 4090",
             "memory_total_gb": 24.0, "compute_units": 1,
             "details": {"gpu_count": 1, "driver_version": "550.90.07"}},
            {"type": "amd", "model": "AMD Instinct MI250",
             "memory_total_gb": 128.0, "compute_units": 220,
             "details": {"gpu_count": 4}},
        ]
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        with patch("sys.stdout"):
            rc = main([])
            # 有 GPU → 应推荐模型 → 返回 0
            assert rc in (0, 1)


class TestIntegrationVerboseMode:
    """verbose 模式集成测试。"""

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_verbose_logs_to_stderr(
        self, mock_container, mock_software, mock_hardware,
    ):
        """verbose 模式输出日志到 stderr。"""
        mock_hardware.return_value = _make_hardware_with_gpu()
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        with patch("sys.stderr") as mock_stderr, \
             patch("sys.stdout"):
            rc = main(["-v"])
            assert rc == 0
            # verbose 模式下 stderr 应有写入
            assert mock_stderr.write.called

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_verbose_with_failure(
        self, mock_container, mock_software, mock_hardware,
    ):
        """verbose 模式下检测失败应打印 traceback 到 stderr。"""
        mock_hardware.side_effect = RuntimeError("hardware crash")
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        with patch("sys.stderr") as mock_stderr, \
             patch("sys.stdout"):
            rc = main(["-v"])
            assert rc == 1
            # verbose + 失败 = stderr 应有输出
            assert mock_stderr.write.called


class TestIntegrationReportContent:
    """报告内容完整性集成测试。"""

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_report_contains_hardware_info(
        self, mock_container, mock_software, mock_hardware,
    ):
        """报告应包含 CPU/GPU 信息。"""
        mock_hardware.return_value = _make_hardware_with_gpu()
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        with patch("sys.stdout") as mock_stdout:
            rc = main([])
            assert rc == 0
            # 检查写入 stdout 的内容
            if mock_stdout.buffer.write.called:
                call_args = mock_stdout.buffer.write.call_args
                written = call_args[0][0].decode("utf-8")
                assert "Intel" in written or "CPU" in written
                assert "A100" in written or "NVIDIA" in written

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_report_mentions_errors(
        self, mock_container, mock_software, mock_hardware,
    ):
        """有错误时报告 stderr 应包含错误信息。"""
        mock_hardware.side_effect = RuntimeError("hardware detection error")
        mock_software.return_value = _make_software()
        mock_container.return_value = _make_container()

        with patch("sys.stderr") as mock_stderr, \
             patch("sys.stdout"):
            rc = main([])
            assert rc == 1
            # stderr 应包含硬件检测失败的相关错误消息
            if mock_stderr.write.called:
                # call_args_list → [call(str1), call(str2), ...]
                # 每个 call 对象解包为 (args, kwargs)
                written = "".join(
                    args[0] for args, _ in mock_stderr.write.call_args_list
                )
                assert "hardware" in written.lower() or "错误" in written
