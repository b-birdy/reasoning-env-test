"""硬件检测器单元测试。"""
import subprocess as sp
import sys
import os
from unittest.mock import MagicMock, patch, mock_open

import pytest

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reasoning_env_test.detectors.hardware import (
    BaseHardwareDetector,
    CPUDetector,
    NvidiaDetector,
    AmdDetector,
    AscendDetector,
    KunlunxinDetector,
    HygonDetector,
    detect_all,
)


# ============================================================
# BaseHardwareDetector
# ============================================================

class TestBaseHardwareDetector:
    def test_cannot_instantiate_abstract(self):
        """抽象基类不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseHardwareDetector()  # type: ignore

    def test_concrete_subclass_works(self):
        """具体子类可以实例化并返回正确格式。"""
        class TestDetector(BaseHardwareDetector):
            def detect(self):
                return self._empty_result("test")

        detector = TestDetector()
        result = detector.detect()
        assert result["type"] == "test"
        assert result["model"] == ""
        assert result["memory_total_gb"] == 0.0
        assert result["compute_units"] == 0
        assert result["details"] == {}


# ============================================================
# CPUDetector
# ============================================================

class TestCPUDetector:
    def test_detect_returns_unified_format(self):
        """CPUDetector.detect() 返回统一格式。"""
        detector = CPUDetector()
        result = detector.detect()

        assert result["type"] == "cpu"
        assert isinstance(result["model"], str)
        assert isinstance(result["memory_total_gb"], (int, float))
        assert isinstance(result["compute_units"], int)
        assert isinstance(result["details"], dict)
        assert "architecture" in result["details"]
        assert "logical_cores" in result["details"]
        assert "os" in result["details"]

    @patch.object(CPUDetector, "_get_cpu_count", return_value=4)
    @patch.object(CPUDetector, "_get_total_memory", return_value=16.0)
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.processor", return_value="")
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.system", return_value="Linux")
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.machine", return_value="x86_64")
    @patch("builtins.open", new_callable=mock_open, read_data="model name : Intel(R) Xeon(R) CPU\n")
    def test_detect_linux_cpuinfo(self, mock_file, mock_machine, mock_system,
                                  mock_processor, mock_memory, mock_cpu_count):
        """Linux 下通过 /proc/cpuinfo 获取 CPU 型号。"""
        detector = CPUDetector()
        result = detector.detect()
        assert result["model"] == "Intel(R) Xeon(R) CPU"
        assert result["compute_units"] == 4
        assert result["details"]["architecture"] == "x86_64"

    @patch.object(CPUDetector, "_get_cpu_count", return_value=8)
    @patch.object(CPUDetector, "_get_total_memory", return_value=32.0)
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.processor", return_value="")
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.system", return_value="Windows")
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.machine", return_value="AMD64")
    @patch("reasoning_env_test.detectors.hardware.cpu.subprocess.check_output")
    def test_detect_windows_wmic(self, mock_check_output, mock_machine, mock_system,
                                 mock_processor, mock_memory, mock_cpu_count):
        """Windows 下通过 wmic 获取 CPU 型号。"""
        mock_check_output.return_value = "Name=Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz\r\n"
        detector = CPUDetector()
        result = detector.detect()
        assert "i7-10700K" in result["model"]
        assert result["compute_units"] == 8

    @patch.object(CPUDetector, "_get_cpu_count", return_value=10)
    @patch.object(CPUDetector, "_get_total_memory", return_value=16.0)
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.processor", return_value="")
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.system", return_value="Darwin")
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.machine", return_value="arm64")
    @patch("reasoning_env_test.detectors.hardware.cpu.subprocess.check_output")
    def test_detect_macos_sysctl(self, mock_check_output, mock_machine, mock_system,
                                 mock_processor, mock_memory, mock_cpu_count):
        """macOS 下通过 sysctl 获取 CPU 型号。"""
        mock_check_output.return_value = "Apple M1 Pro\n"
        detector = CPUDetector()
        result = detector.detect()
        assert "M1" in result["model"]

    @patch.object(CPUDetector, "_get_cpu_count", return_value=4)
    @patch.object(CPUDetector, "_get_total_memory", return_value=16.0)
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.processor", return_value="")
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.system", return_value="Linux")
    @patch("reasoning_env_test.detectors.hardware.cpu.platform.machine", return_value="x86_64")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_detect_no_model_info(self, mock_file, mock_machine, mock_system,
                                  mock_processor, mock_memory, mock_cpu_count):
        """没有 CPU 型号信息时返回空字符串。"""
        detector = CPUDetector()
        result = detector.detect()
        assert result["model"] == ""
        assert result["compute_units"] == 4


# ============================================================
# NvidiaDetector
# ============================================================

class TestNvidiaDetector:
    def test_detect_no_nvidia_hardware(self):
        """无 NVIDIA 硬件时返回空结果。"""
        with patch("reasoning_env_test.detectors.hardware.nvidia.shutil.which", return_value=None):
            detector = NvidiaDetector()
            result = detector.detect()
            assert result["type"] == "nvidia"
            assert result["model"] == ""
            assert result["memory_total_gb"] == 0.0
            assert result["compute_units"] == 0

    @patch("reasoning_env_test.detectors.hardware.nvidia.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("reasoning_env_test.detectors.hardware.nvidia.subprocess.check_output")
    def test_detect_via_nvidia_smi(self, mock_check_output, mock_which):
        """通过 nvidia-smi 命令检测 NVIDIA GPU。"""
        mock_check_output.return_value = (
            "0, NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0\n"
            "1, NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0\n"
        )
        detector = NvidiaDetector()
        result = detector.detect()
        assert result["type"] == "nvidia"
        assert "A100" in result["model"]
        assert result["memory_total_gb"] == 80.0
        assert result["details"]["gpu_count"] == 2

    @patch("reasoning_env_test.detectors.hardware.nvidia.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("reasoning_env_test.detectors.hardware.nvidia.subprocess.check_output", side_effect=FileNotFoundError)
    def test_nvidia_smi_fails_gracefully(self, mock_check_output, mock_which):
        """nvidia-smi 命令失败时返回空结果。"""
        detector = NvidiaDetector()
        result = detector.detect()
        assert result["type"] == "nvidia"
        assert result["model"] == ""

    def test_detect_via_pynvml(self):
        """通过 pynvml 库检测 NVIDIA GPU。"""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1

        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA RTX 4090"

        mock_mem = MagicMock()
        mock_mem.total = 24 * 1024 ** 3  # 24 GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch("reasoning_env_test.detectors.hardware.nvidia.shutil.which", return_value=None):
                detector = NvidiaDetector()
                result = detector.detect()
                assert result["type"] == "nvidia"
                assert "4090" in result["model"]
                assert result["memory_total_gb"] == 24.0


# ============================================================
# AmdDetector
# ============================================================

class TestAmdDetector:
    def test_detect_no_amd_hardware(self):
        """无 AMD ROCm 时返回空结果。"""
        with patch("reasoning_env_test.detectors.hardware.amd.shutil.which", return_value=None):
            detector = AmdDetector()
            result = detector.detect()
            assert result["type"] == "amd"
            assert result["model"] == ""
            assert result["memory_total_gb"] == 0.0

    @patch("reasoning_env_test.detectors.hardware.amd.shutil.which", return_value="/opt/rocm/bin/rocm-smi")
    @patch("reasoning_env_test.detectors.hardware.amd.subprocess.check_output")
    def test_detect_rocm_smi(self, mock_check_output, mock_which):
        """通过 rocm-smi 命令检测 AMD GPU。"""
        mock_output = """ROCm System Management Interface
=================================== GPU 0 ===================================
Card model:                       AMD Instinct MI250X
VRAM:                             65536 MB
Compute Units:                    220
=================================== GPU 1 ===================================
Card model:                       AMD Instinct MI250X
VRAM:                             65536 MB
Compute Units:                    220
"""
        mock_check_output.return_value = mock_output
        detector = AmdDetector()
        result = detector.detect()
        assert result["type"] == "amd"
        assert "MI250X" in result["model"]
        assert result["details"]["gpu_count"] == 2
        assert result["memory_total_gb"] == 64.0  # 65536 MB = 64 GB

    @patch("reasoning_env_test.detectors.hardware.amd.shutil.which", return_value="/opt/rocm/bin/rocm-smi")
    @patch("reasoning_env_test.detectors.hardware.amd.subprocess.check_output", side_effect=FileNotFoundError)
    def test_rocm_smi_fails_gracefully(self, mock_check_output, mock_which):
        """rocm-smi 命令失败时返回空结果。"""
        detector = AmdDetector()
        result = detector.detect()
        assert result["type"] == "amd"
        assert result["model"] == ""


# ============================================================
# AscendDetector
# ============================================================

class TestAscendDetector:
    def test_detect_no_ascend_hardware(self):
        """无昇腾 NPU 时返回空结果。"""
        with patch("reasoning_env_test.detectors.hardware.ascend.shutil.which", return_value=None):
            detector = AscendDetector()
            result = detector.detect()
            assert result["type"] == "ascend"
            assert result["model"] == ""
            assert result["memory_total_gb"] == 0.0

    @patch("reasoning_env_test.detectors.hardware.ascend.shutil.which", return_value="/usr/local/bin/npu-smi")
    @patch("reasoning_env_test.detectors.hardware.ascend.subprocess.check_output")
    def test_detect_npu_smi(self, mock_check_output, mock_which):
        """通过 npu-smi 命令检测昇腾 NPU。"""
        mock_output = """NPU ID: 0
Product Name: Ascend 910B
Memory: 32768 MB Total
NPU ID: 1
Product Name: Ascend 910B
Memory: 32768 MB Total
"""
        mock_check_output.return_value = mock_output
        detector = AscendDetector()
        result = detector.detect()
        assert result["type"] == "ascend"
        assert "910" in result["model"]
        assert result["details"]["npu_count"] == 2
        assert result["memory_total_gb"] == 32.0  # 32768 MB = 32 GB

    @patch("reasoning_env_test.detectors.hardware.ascend.shutil.which", return_value="/usr/local/bin/npu-smi")
    @patch("reasoning_env_test.detectors.hardware.ascend.subprocess.check_output", side_effect=sp.CalledProcessError(1, "npu-smi"))
    def test_npu_smi_fails_gracefully(self, mock_check_output, mock_which):
        """npu-smi 命令失败时返回空结果。"""
        detector = AscendDetector()
        result = detector.detect()
        assert result["type"] == "ascend"
        assert result["model"] == ""


# ============================================================
# KunlunxinDetector
# ============================================================

class TestKunlunxinDetector:
    def test_detect_no_kunlunxin_hardware(self):
        """无昆仑芯 XPU 时返回空结果。"""
        with patch("reasoning_env_test.detectors.hardware.kunlunxin.shutil.which", return_value=None):
            detector = KunlunxinDetector()
            result = detector.detect()
            assert result["type"] == "kunlunxin"
            assert result["model"] == ""
            assert result["memory_total_gb"] == 0.0

    @patch("reasoning_env_test.detectors.hardware.kunlunxin.shutil.which", return_value="/usr/bin/xpu-smi")
    def test_detect_xpu_smi_found(self, mock_which):
        """检测到 xpu-smi 命令时返回占位信息。"""
        detector = KunlunxinDetector()
        result = detector.detect()
        assert result["type"] == "kunlunxin"
        assert "placeholder" in result["model"].lower()
        assert result["details"].get("command_found") == "xpu-smi"


# ============================================================
# HygonDetector
# ============================================================

class TestHygonDetector:
    def test_detect_no_hygon_hardware(self):
        """无海光 DCU 时返回空结果。"""
        with patch("reasoning_env_test.detectors.hardware.hygon.shutil.which", return_value=None):
            detector = HygonDetector()
            result = detector.detect()
            assert result["type"] == "hygon"
            assert result["model"] == ""
            assert result["memory_total_gb"] == 0.0

    @patch("reasoning_env_test.detectors.hardware.hygon.shutil.which")
    def test_detect_hygon_smi_found(self, mock_which):
        """检测到 hygon-smi 命令时返回占位信息。"""
        mock_which.side_effect = lambda x: "/usr/bin/" + x if x == "hygon-smi" else None
        detector = HygonDetector()
        result = detector.detect()
        assert result["type"] == "hygon"
        assert "placeholder" in result["model"].lower()
        assert "hygon-smi" in result["details"].get("commands_found", [])

    @patch("reasoning_env_test.detectors.hardware.hygon.shutil.which")
    def test_detect_dcu_smi_found(self, mock_which):
        """检测到 dcu-smi 命令时返回占位信息。"""
        mock_which.side_effect = lambda x: "/usr/bin/" + x if x == "dcu-smi" else None
        detector = HygonDetector()
        result = detector.detect()
        assert result["type"] == "hygon"
        assert "placeholder" in result["model"].lower()
        assert "dcu-smi" in result["details"].get("commands_found", [])


# ============================================================
# detect_all
# ============================================================

class TestDetectAll:
    def test_detect_all_returns_list(self):
        """detect_all() 返回包含所有检测器结果的列表。"""
        results = detect_all()
        assert isinstance(results, list)
        assert len(results) == 6

        types = {r["type"] for r in results}
        expected_types = {"cpu", "nvidia", "amd", "ascend", "kunlunxin", "hygon"}
        assert types == expected_types

    def test_all_results_have_unified_format(self):
        """每个检测结果都遵循统一输出格式。"""
        results = detect_all()
        required_keys = {"type", "model", "memory_total_gb", "compute_units", "details"}
        for result in results:
            assert required_keys.issubset(result.keys()), (
                f"{result['type']} 缺少字段: {required_keys - result.keys()}"
            )
            assert isinstance(result["type"], str)
            assert isinstance(result["model"], str)
            assert isinstance(result["memory_total_gb"], (int, float))
            assert isinstance(result["compute_units"], int)
            assert isinstance(result["details"], dict)
