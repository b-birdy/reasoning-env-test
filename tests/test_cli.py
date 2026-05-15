"""命令行入口单元测试。"""
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reasoning_env_test.__main__ import parse_args, main


class TestParseArgs:
    """parse_args 函数测试。"""

    def test_default_args(self):
        """无参数时使用默认值。"""
        args = parse_args([])
        assert args.output is None
        assert args.verbose is False

    def test_output_short(self):
        """-o 参数。"""
        args = parse_args(["-o", "report.md"])
        assert args.output == "report.md"
        assert args.verbose is False

    def test_output_long(self):
        """--output 参数。"""
        args = parse_args(["--output", "report.md"])
        assert args.output == "report.md"

    def test_verbose_short(self):
        """-v 参数。"""
        args = parse_args(["-v"])
        assert args.verbose is True
        assert args.output is None

    def test_verbose_long(self):
        """--verbose 参数不存在（只有 -v）。"""
        # argparse 中 verbose 没有长选项，只支持 -v
        pass

    def test_output_and_verbose(self):
        """同时指定 -o 和 -v。"""
        args = parse_args(["-o", "out.md", "-v"])
        assert args.output == "out.md"
        assert args.verbose is True

    def test_help(self):
        """--help 应打印帮助信息并退出。"""
        with pytest.raises(SystemExit):
            parse_args(["--help"])


class TestMainFunction:
    """main 函数整体流程测试。"""

    def _make_hardware(self):
        return [
            {"type": "cpu", "model": "Test CPU", "memory_total_gb": 16.0,
             "compute_units": 8, "details": {"architecture": "x86_64"}},
        ]

    def _make_software(self):
        return {
            "commands": {"python3": True, "nvidia-smi": False},
            "frameworks": {"torch": "2.0.0"},
            "python_version": "3.10.0",
            "python_ok": True,
            "cuda_version": None,
            "rocm_version": None,
        }

    def _make_container(self):
        return {
            "in_container": False, "container_type": None,
            "k8s_pod": False, "docker_installed": True,
            "memory_limit_bytes": None, "cpu_quota": None,
        }

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_main_success(self, mock_container, mock_software, mock_hardware):
        """成功路径：所有检测正常，输出到 stdout。"""
        mock_hardware.return_value = self._make_hardware()
        mock_software.return_value = self._make_software()
        mock_container.return_value = self._make_container()

        with patch("sys.stdout") as mock_stdout:
            rc = main(["-v"])
            assert rc == 0, f"main() returned {rc}, expected 0"
            # stdout.buffer.write 应被调用（报告输出）
            assert mock_stdout.buffer.write.called

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_main_hardware_failure(
        self, mock_container, mock_software, mock_hardware,
    ):
        """硬件检测失败时，其余步骤应继续执行（降级为默认值）。"""
        mock_hardware.side_effect = RuntimeError("no hardware")
        mock_software.return_value = {}
        mock_container.return_value = {}

        with patch("sys.stdout"):
            rc = main([])
            # 硬件失败，但报告仍可生成（空数据）
            assert rc == 1, "硬件失败时 main() 应返回 1"

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_output_to_file(
        self, mock_container, mock_software, mock_hardware, tmp_path,
    ):
        """指定 --output 时写入文件。"""
        output_file = tmp_path / "report.md"

        mock_hardware.return_value = [
            {"type": "cpu", "model": "Test CPU", "memory_total_gb": 16.0,
             "compute_units": 8, "details": {"architecture": "x86_64"}},
        ]
        mock_software.return_value = {
            "commands": {"python3": True},
            "frameworks": {},
            "python_version": "3.10.0",
            "python_ok": True,
            "cuda_version": None,
            "rocm_version": None,
        }
        mock_container.return_value = {
            "in_container": False, "container_type": None,
            "k8s_pod": False, "docker_installed": False,
            "memory_limit_bytes": None, "cpu_quota": None,
        }

        rc = main(["-o", str(output_file)])
        assert rc == 0

        # 确认文件已写入且内容包含报告标题
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "推理环境检测报告" in content

    @patch("reasoning_env_test.detectors.hardware.detect_all")
    @patch("reasoning_env_test.detectors.software.software.detect_all")
    @patch("reasoning_env_test.detectors.container.container.detect_all")
    def test_verbose_output(
        self, mock_container, mock_software, mock_hardware,
    ):
        """--verbose 模式应将日志输出到 stderr。"""
        mock_hardware.return_value = [
            {"type": "cpu", "model": "Test CPU", "memory_total_gb": 16.0,
             "compute_units": 8, "details": {"architecture": "x86_64"}},
        ]
        mock_software.return_value = self._make_software()
        mock_container.return_value = self._make_container()

        with patch("sys.stderr") as mock_stderr, \
             patch("sys.stdout"):
            rc = main(["-v"])
            assert rc == 0
            # verbose 模式下 stderr 应有输出
            assert mock_stderr.write.called
