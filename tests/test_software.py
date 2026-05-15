"""单元测试：software.py 检测模块"""

import sys
import os
from unittest.mock import patch, MagicMock, mock_open

import pytest

from reasoning_env_test.detectors.software.software import (
    BIN_DIRS,
    COMMANDS,
    FRAMEWORKS,
    scan_bin_dirs,
    detect_commands,
    detect_frameworks,
    detect_python,
    detect_cuda,
    detect_rocm,
    detect_all,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def reset_mocks():
    """每个测试后自动清理 mock 状态。"""
    yield


# ============================================================
# scan_bin_dirs
# ============================================================


class TestScanBinDirs:
    def test_linux_scan_existing_dirs(self):
        """Linux 下扫描存在的 bin 目录，应返回可执行文件列表。"""
        with (
            patch("reasoning_env_test.detectors.software.software.sys.platform", "linux"),
            patch("os.path.isdir", return_value=True),
            patch("os.listdir", return_value=["python3", "ls", "nvidia-smi"]),
            patch("os.access", return_value=True),
        ):
            result = scan_bin_dirs()
            # 3 个目录 x 3 个文件 = 9 条
            assert len(result) == 9
            assert all(f.endswith(("python3", "ls", "nvidia-smi")) for f in result)

    def test_linux_no_bin_dirs(self):
        """Linux 下 bin 目录不存在时应返回空列表。"""
        with (
            patch("reasoning_env_test.detectors.software.software.sys.platform", "linux"),
            patch("os.path.isdir", return_value=False),
        ):
            result = scan_bin_dirs()
            assert result == []

    def test_non_linux_scan_path(self):
        """非 Linux 下降级为扫描 PATH 目录。"""
        with (
            patch("reasoning_env_test.detectors.software.software.sys.platform", "win32"),
            patch.dict(os.environ, {"PATH": "C:\\windows;C:\\tools"}, clear=True),
            patch("os.path.isdir", return_value=True),
            patch("os.listdir", side_effect=[["python.exe", "docker.exe"], ["kubectl.exe"]]),
            patch("os.access", return_value=True),
        ):
            result = scan_bin_dirs()
            assert len(result) == 3

    def test_non_linux_no_path_dirs(self):
        """非 Linux 下 PATH 目录均不存在时应返回空列表。"""
        with (
            patch("reasoning_env_test.detectors.software.software.sys.platform", "win32"),
            patch("os.path.isdir", return_value=False),
        ):
            result = scan_bin_dirs()
            assert result == []

    def test_linux_permission_error_skipped(self):
        """PermissionError 应被静默跳过。"""
        with (
            patch("reasoning_env_test.detectors.software.software.sys.platform", "linux"),
            patch("os.path.isdir", return_value=True),
            patch("os.listdir", side_effect=PermissionError("denied")),
        ):
            result = scan_bin_dirs()
            assert result == []


# ============================================================
# detect_commands
# ============================================================


class TestDetectCommands:
    def test_all_commands_found(self):
        """所有命令都存在时应全部返回 True。"""
        with patch("shutil.which", return_value="/usr/bin/python3"):
            result = detect_commands()
            for cmd in COMMANDS:
                assert result[cmd] is True, f"{cmd} should be True"

    def test_all_commands_missing(self):
        """所有命令都不存在时应全部返回 False。"""
        with patch("shutil.which", return_value=None):
            result = detect_commands()
            for cmd in COMMANDS:
                assert result[cmd] is False, f"{cmd} should be False"

    def test_partial_commands(self):
        """部分命令存在时只返回对应 True。"""

        def fake_which(cmd, **kwargs):
            return "/usr/bin/" + cmd if cmd in ("python", "docker") else None

        with patch("shutil.which", side_effect=fake_which):
            result = detect_commands()
            assert result["python"] is True
            assert result["docker"] is True
            assert result["kubectl"] is False
            assert result["nvidia-smi"] is False

    def test_result_keys_match_commands_list(self):
        """返回的 keys 应严格等于 COMMANDS 列表。"""
        with patch("shutil.which", return_value=None):
            result = detect_commands()
            assert set(result.keys()) == set(COMMANDS)


# ============================================================
# detect_frameworks
# ============================================================


class TestDetectFrameworks:
    def test_all_frameworks_via_pip(self):
        """所有框架通过 pip 包检测到。"""
        fake_pkgs = {
            "vllm": "0.6.0",
            "text-generation": "2.0.0",
            "ollama": "0.3.0",
            "tensorrt_llm": "0.12.0",
            "onnxruntime": "1.18.0",
            "torch": "2.5.0",
            "tensorflow": "2.17.0",
        }
        with patch(
            "reasoning_env_test.detectors.software.software._get_pip_packages",
            return_value=fake_pkgs,
        ):
            result = detect_frameworks()
            for key, ver in FRAMEWORKS.items():
                expected_key = ver  # FRAMEWORKS value is the output key
                assert result[expected_key] == fake_pkgs[key], f"{expected_key} mismatch"

    def test_no_frameworks_installed(self):
        """没有框架安装时应全部返回 None。"""
        with (
            patch(
                "reasoning_env_test.detectors.software.software._get_pip_packages",
                return_value={},
            ),
            patch("shutil.which", return_value=None),
        ):
            result = detect_frameworks()
            for key in FRAMEWORKS.values():
                assert result[key] is None, f"{key} should be None"

    def test_framework_detected_via_which_fallback(self):
        """pip 检测不到时回退到 which --version。"""
        with (
            patch(
                "reasoning_env_test.detectors.software.software._get_pip_packages",
                return_value={},
            ),
            patch("shutil.which", return_value="/usr/local/bin/ollama"),
            patch(
                "subprocess.run",
                return_value=MagicMock(
                    stdout="ollama version 0.3.0\n", stderr="", returncode=0
                ),
            ),
        ):
            result = detect_frameworks()
            assert result["ollama"] == "ollama version 0.3.0"

    def test_which_fallback_fails_gracefully(self):
        """which 回退也失败时应返回 None 而非抛异常。"""
        with (
            patch(
                "reasoning_env_test.detectors.software.software._get_pip_packages",
                return_value={},
            ),
            patch("shutil.which", return_value="/usr/bin/some_cmd"),
            patch("subprocess.run", side_effect=OSError("not executable")),
        ):
            result = detect_frameworks()
            for key in FRAMEWORKS.values():
                assert result[key] is None


# ============================================================
# detect_python
# ============================================================


class TestDetectPython:
    def _make_version_info(self, major, minor, micro):
        """创建一个类似 sys.version_info 的 named tuple。"""
        import collections
        VersionInfo = collections.namedtuple(
            "version_info", ["major", "minor", "micro", "releaselevel", "serial"]
        )
        return VersionInfo(major, minor, micro, "final", 0)

    def test_python_3_8(self):
        """Python 3.8 应通过检查。"""
        with patch.object(sys, "version_info", self._make_version_info(3, 8, 0)):
            ver, ok = detect_python()
            assert ver == "3.8.0"
            assert ok is True

    def test_python_3_12(self):
        """Python 3.12 应通过检查。"""
        with patch.object(sys, "version_info", self._make_version_info(3, 12, 5)):
            ver, ok = detect_python()
            assert ver == "3.12.5"
            assert ok is True

    def test_python_3_7_fails(self):
        """Python 3.7 应不通过检查。"""
        with patch.object(sys, "version_info", self._make_version_info(3, 7, 9)):
            ver, ok = detect_python()
            assert ver == "3.7.9"
            assert ok is False

    def test_python_2_7_fails(self):
        """Python 2.7 应不通过检查。"""
        with patch.object(sys, "version_info", self._make_version_info(2, 7, 18)):
            ver, ok = detect_python()
            assert ver == "2.7.18"
            assert ok is False


# ============================================================
# detect_cuda
# ============================================================


class TestDetectCuda:
    def test_from_nvidia_smi(self):
        """从 nvidia-smi 输出解析 CUDA 版本。"""
        nvidia_smi_output = """\
Tue May 15 10:00:00 2026
+------------------------------------------------------+
| NVIDIA-SMI 550.90.07  Driver Version: 550.90.07  CUDA Version: 12.4 |
+------------------------------------------------------+
"""
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "subprocess.run",
                return_value=MagicMock(stdout=nvidia_smi_output, stderr="", returncode=0),
            ),
        ):
            ver = detect_cuda()
            assert ver == "12.4"

    def test_from_nvcc(self):
        """nvidia-smi 失败时从 nvcc --version 解析。"""
        nvcc_output = """\
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jan_12_12:00:00_PST_2024
Cuda compilation tools, release 11.8, V11.8.89
"""
        # nvidia-smi 存在但无法解析 → nvcc 回退
        with (
            patch(
                "shutil.which",
                side_effect=lambda cmd, **kw: "/usr/bin/" + cmd if cmd in ("nvidia-smi", "nvcc") else None,
            ),
            patch(
                "subprocess.run",
                side_effect=[
                    MagicMock(stdout="", stderr="Error: no GPU", returncode=1),
                    MagicMock(stdout=nvcc_output, stderr="", returncode=0),
                ],
            ),
        ):
            ver = detect_cuda()
            assert ver == "11.8"

    def test_no_cuda_tools(self):
        """没有任何 CUDA 工具时返回 None。"""
        with patch("shutil.which", return_value=None):
            ver = detect_cuda()
            assert ver is None

    def test_nvidia_smi_exception_handled(self):
        """nvidia-smi 执行异常应安静回退。"""
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", side_effect=FileNotFoundError("not found")),
        ):
            ver = detect_cuda()
            assert ver is None


# ============================================================
# detect_rocm
# ============================================================


class TestDetectRocm:
    def test_from_rocm_smi(self):
        """从 rocm-smi --showversion 解析 ROCm 版本。"""
        rocm_smi_output = "ROCm Version: 6.1.2 \n"
        with (
            patch("shutil.which", return_value="/usr/bin/rocm-smi"),
            patch(
                "subprocess.run",
                return_value=MagicMock(stdout=rocm_smi_output, stderr="", returncode=0),
            ),
        ):
            ver = detect_rocm()
            assert ver == "6.1.2"

    def test_from_opt_rocm_version_file(self):
        """rocm-smi 失败时从 /opt/rocm/version 文件读取。"""
        with (
            patch("shutil.which", return_value=None),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="6.0.0\n")),
        ):
            ver = detect_rocm()
            assert ver == "6.0.0"

    def test_no_rocm_tools(self):
        """没有任何 ROCm 工具时返回 None。"""
        with (
            patch("shutil.which", return_value=None),
            patch("os.path.exists", return_value=False),
        ):
            ver = detect_rocm()
            assert ver is None

    def test_rocm_smi_exception_handled(self):
        """rocm-smi 执行异常应安静回退到文件。"""
        with (
            patch("shutil.which", return_value="/usr/bin/rocm-smi"),
            patch("subprocess.run", side_effect=PermissionError("denied")),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="6.1.0\n")),
        ):
            ver = detect_rocm()
            assert ver == "6.1.0"

    def test_rocm_smi_no_version_in_output(self):
        """rocm-smi 输出无版本号时应回退到文件。"""
        with (
            patch("shutil.which", return_value="/usr/bin/rocm-smi"),
            patch(
                "subprocess.run",
                return_value=MagicMock(stdout="no version here\n", stderr="", returncode=0),
            ),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="5.7.1\n")),
        ):
            ver = detect_rocm()
            assert ver == "5.7.1"


# ============================================================
# detect_all
# ============================================================


class TestDetectAll:
    def test_returns_all_required_keys(self):
        """detect_all 返回的字典应包含全部必需的 key。"""
        required_keys = {
            "commands",
            "frameworks",
            "python_version",
            "python_ok",
            "cuda_version",
            "rocm_version",
        }

        with patch.multiple(
            "reasoning_env_test.detectors.software.software",
            detect_commands=MagicMock(return_value={c: False for c in COMMANDS}),
            detect_frameworks=MagicMock(
                return_value={v: None for v in FRAMEWORKS.values()}
            ),
            detect_python=MagicMock(return_value=("3.12.0", True)),
            detect_cuda=MagicMock(return_value=None),
            detect_rocm=MagicMock(return_value=None),
        ):
            result = detect_all()
            assert set(result.keys()) == required_keys

    def test_values_are_correct_types(self):
        """返回值应具有正确的类型。"""
        with patch.multiple(
            "reasoning_env_test.detectors.software.software",
            detect_commands=MagicMock(return_value={c: False for c in COMMANDS}),
            detect_frameworks=MagicMock(
                return_value={v: None for v in FRAMEWORKS.values()}
            ),
            detect_python=MagicMock(return_value=("3.12.0", True)),
            detect_cuda=MagicMock(return_value="12.4"),
            detect_rocm=MagicMock(return_value="6.1.0"),
        ):
            result = detect_all()
            assert isinstance(result["commands"], dict)
            assert isinstance(result["frameworks"], dict)
            assert isinstance(result["python_version"], str)
            assert isinstance(result["python_ok"], bool)
            assert isinstance(result["cuda_version"], (str, type(None)))
            assert isinstance(result["rocm_version"], (str, type(None)))
