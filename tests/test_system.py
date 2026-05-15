"""系统详情检测单元测试。"""
import os
import platform
import socket
import subprocess
import sys
from unittest.mock import MagicMock, PropertyMock, mock_open, patch

import pytest

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reasoning_env_test.detectors.system import (
    detect_all,
    detect_cpu,
    detect_memory,
    detect_os,
)


# ============================================================
# 测试辅助数据
# ============================================================

SAMPLE_OS_RELEASE = """NAME="Ubuntu"
VERSION="22.04.3 LTS"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 22.04.3 LTS"
VERSION_ID="22.04"
"""

SAMPLE_CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 143
model name\t: Intel(R) Xeon(R) Platinum 8468
stepping\t: 8
cpu MHz\t\t: 3000.000
cache size\t: 122880 KB
physical id\t: 0
siblings\t: 96
core id\t\t: 0
cpu cores\t: 48
apicid\t\t: 0
initial apicid\t: 0
fpu\t\t: yes
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts pku ospke avx512_vnmi md_clear flush_l1d arch_capabilities

processor\t: 1
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 143
model name\t: Intel(R) Xeon(R) Platinum 8468
stepping\t: 8
cpu MHz\t\t: 3000.000
cache size\t: 122880 KB
physical id\t: 0
siblings\t: 96
core id\t\t: 1
cpu cores\t: 48
apicid\t\t: 1
initial apicid\t: 1
fpu\t\t: yes
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts pku ospke avx512_vnmi md_clear flush_l1d arch_capabilities

processor\t: 2
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 143
model name\t: Intel(R) Xeon(R) Platinum 8468
stepping\t: 8
cpu MHz\t\t: 3000.000
cache size\t: 122880 KB
physical id\t: 1
siblings\t: 96
core id\t\t: 0
cpu cores\t: 48
apicid\t\t: 48
initial apicid\t: 48
fpu\t\t: yes
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts pku ospke avx512_vnmi md_clear flush_l1d arch_capabilities

processor\t: 3
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 143
model name\t: Intel(R) Xeon(R) Platinum 8468
stepping\t: 8
cpu MHz\t\t: 3000.000
cache size\t: 122880 KB
physical id\t: 1
siblings\t: 96
core id\t\t: 1
cpu cores\t: 48
apicid\t\t: 49
initial apicid\t: 49
fpu\t\t: yes
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts pku ospke avx512_vnmi md_clear flush_l1d arch_capabilities
"""

SAMPLE_MEMINFO = "MemTotal: 1583296740 kB\n"

SAMPLE_DMIDECODE = """# dmidecode 3.5
Getting SMBIOS data from sysfs.
SMBIOS 3.0.0 present.

Handle 0x0039, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tError Correction Type: Multi-bit ECC
\tMaximum Capacity: 6 TB
\tError Information Handle: Not Provided
\tNumber Of Devices: 32

Handle 0x0042, DMI type 17, 92 bytes
Memory Device
\tArray Handle: 0x0039
\tError Information Handle: Not Provided
\tTotal Width: 72 bits
\tData Width: 64 bits
\tSize: 96 GB
\tForm Factor: DIMM
\tSet: None
\tLocator: CPU1_DIMM_A1
\tBank Locator: NODE0
\tType: DDR5
\tType Detail: Synchronous Unbuffered (Unbuffered)
\tSpeed: 4800 MT/s
\tManufacturer: Samsung
\tSerial Number: 12345678
\tAsset Tag: Unknown
\tPart Number: M393B1K70DH0-YH9

Handle 0x0043, DMI type 17, 92 bytes
Memory Device
\tArray Handle: 0x0039
\tError Information Handle: Not Provided
\tTotal Width: 72 bits
\tData Width: 64 bits
\tSize: 96 GB
\tForm Factor: DIMM
\tSet: None
\tLocator: CPU1_DIMM_A2
\tBank Locator: NODE0
\tType: DDR5
\tType Detail: Synchronous Unbuffered (Unbuffered)
\tSpeed: 4800 MT/s
\tManufacturer: Samsung
\tSerial Number: 87654321
\tAsset Tag: Unknown
\tPart Number: M393B1K70DH0-YH9
"""


# ============================================================
# TestDetectOS
# ============================================================


class TestDetectOS:
    def _mock_uname(self, kernel="5.15.0-91-generic"):
        """Helper: mock platform.uname() to return a namedtuple with .release."""
        from collections import namedtuple
        uname_info = namedtuple("uname_result", ["release"])
        return patch.object(platform, "uname", return_value=uname_info(release=kernel))

    def test_detect_os_linux_with_os_release(self):
        """Linux 下通过 /etc/os-release 获取 OS 信息。"""
        with patch.object(platform, "system", return_value="Linux"):
            with patch("builtins.open", mock_open(read_data=SAMPLE_OS_RELEASE)):
                with patch.object(socket, "gethostname", return_value="server-01"):
                    with patch.object(platform, "machine", return_value="x86_64"):
                        with self._mock_uname("5.15.0-91-generic"):
                            result = detect_os()

        assert result["distribution"] == "ubuntu"
        assert result["version"] == "22.04"
        assert result["kernel"] == "5.15.0-91-generic"
        assert result["hostname"] == "server-01"
        assert result["architecture"] == "x86_64"

    def test_detect_os_file_not_found(self):
        """/etc/os-release 不存在时返回 Unknown。"""
        with patch.object(platform, "system", return_value="Linux"):
            with patch("builtins.open", side_effect=FileNotFoundError):
                with patch.object(socket, "gethostname", return_value="server-01"):
                    with patch.object(platform, "machine", return_value="x86_64"):
                        with self._mock_uname("5.15.0-91-generic"):
                            result = detect_os()

        assert result["distribution"] == "Unknown"
        assert result["kernel"] == "5.15.0-91-generic"

    def test_detect_os_non_linux(self):
        """非 Linux 系统不读 /etc/os-release。"""
        with patch.object(platform, "system", return_value="Windows"):
            with patch.object(socket, "gethostname", return_value="WIN-PC"):
                with patch.object(platform, "machine", return_value="AMD64"):
                    with self._mock_uname("10.0.19045"):
                        result = detect_os()

        assert result["hostname"] == "WIN-PC"
        assert result["architecture"] == "AMD64"
        assert result["distribution"] == "Unknown"

    def test_detect_os_exception_handling(self):
        """所有异常被捕获，不抛错。"""
        with patch.object(platform, "system", side_effect=Exception("mock error")):
            with patch.object(socket, "gethostname", side_effect=Exception("socket error")):
                with self._mock_uname(""):
                    result = detect_os()

        assert result["distribution"] == "Unknown"
        assert result["hostname"] == "Unknown"
        assert result["kernel"] == "Unknown"


# ============================================================
# TestDetectCPU
# ============================================================


class TestDetectCPU:
    def test_detect_cpu_full_info(self):
        """完整的 /proc/cpuinfo 解析。"""
        with patch.object(platform, "system", return_value="Linux"):
            with patch.object(platform, "machine", return_value="x86_64"):
                with patch("builtins.open", mock_open(read_data=SAMPLE_CPUINFO)):
                    with patch("os.path.exists", return_value=False):
                        with patch("os.listdir", return_value=[]):
                            result = detect_cpu()

        assert result["model_name"] == "Intel(R) Xeon(R) Platinum 8468"
        assert result["architecture"] == "x86_64"
        assert result["physical_cores"] == 96  # 48 cores * 2 sockets
        assert result["logical_cores"] == 4   # only 4 entries in sample
        assert result["frequency_mhz"] == 3000.0
        assert result["sockets"] == 2
        assert result["threads_per_core"] == 0  # 4 / 96 = 0 with truncating int
        assert len(result["flags"]) > 10
        assert "avx512f" in result["flags"]
        assert "avx2" in result["flags"]

    def test_detect_cpu_no_cpuinfo(self):
        """/proc/cpuinfo 不存在时返回默认值。"""
        with patch.object(platform, "system", return_value="Linux"):
            with patch.object(platform, "machine", return_value="x86_64"):
                with patch("builtins.open", side_effect=FileNotFoundError):
                    result = detect_cpu()

        assert result["model_name"] == "Unknown"
        assert result["physical_cores"] == 0
        assert result["logical_cores"] == 0

    def test_detect_cpu_non_linux(self):
        """非 Linux 系统返回默认值。"""
        with patch.object(platform, "system", return_value="Windows"):
            with patch.object(platform, "machine", return_value="AMD64"):
                result = detect_cpu()

        assert result["architecture"] == "AMD64"
        assert result["model_name"] == "Unknown"
        assert result["physical_cores"] == 0

    def test_detect_cpu_with_sys_cache(self):
        """从 /sys/.../cache 检测缓存。"""

        from reasoning_env_test.detectors.system.system_info import (
            _detect_cache_info,
            _parse_cpuinfo,
        )

        with patch.object(platform, "system", return_value="Linux"):
            with patch("builtins.open") as mock_open_obj:
                def _open_side(path, *args, **kwargs):
                    if path == "/proc/cpuinfo":
                        return mock_open(read_data=SAMPLE_CPUINFO).return_value
                    raise FileNotFoundError(f"Mock not found: {path}")
                mock_open_obj.side_effect = _open_side
                processors = _parse_cpuinfo()

        assert len(processors) > 0

        sysfs_values = {
            "/sys/devices/system/cpu/cpu0/cache/index0/type": "Data",
            "/sys/devices/system/cpu/cpu0/cache/index0/size": "48K",
            "/sys/devices/system/cpu/cpu0/cache/index0/level": "1",
            "/sys/devices/system/cpu/cpu0/cache/index1/type": "Instruction",
            "/sys/devices/system/cpu/cpu0/cache/index1/size": "32K",
            "/sys/devices/system/cpu/cpu0/cache/index1/level": "1",
            "/sys/devices/system/cpu/cpu0/cache/index2/type": "Unified",
            "/sys/devices/system/cpu/cpu0/cache/index2/size": "2M",
            "/sys/devices/system/cpu/cpu0/cache/index2/level": "2",
            "/sys/devices/system/cpu/cpu0/cache/index3/type": "Unified",
            "/sys/devices/system/cpu/cpu0/cache/index3/size": "120M",
            "/sys/devices/system/cpu/cpu0/cache/index3/level": "3",
        }

        with patch("builtins.open") as mock_open_obj:
            def _open_side_effect(path, *args, **kwargs):
                # Normalize path separators (os.path.join on Windows uses \)
                norm_path = path.replace(os.sep, "/")
                if norm_path in sysfs_values:
                    return mock_open(read_data=sysfs_values[norm_path]).return_value
                raise FileNotFoundError(f"Mock not found: {path}")
            mock_open_obj.side_effect = _open_side_effect

            with patch("os.path.isdir", return_value=True):
                with patch("os.listdir", return_value=["index0", "index1", "index2", "index3"]):
                    caches = _detect_cache_info(processors)

        assert caches["cache_l1d"] == "48K", f"Got l1d={caches['cache_l1d']}"
        assert caches["cache_l1i"] == "32K", f"Got l1i={caches['cache_l1i']}"
        assert caches["cache_l2"] == "2M", f"Got l2={caches['cache_l2']}"
        assert caches["cache_l3"] == "120M", f"Got l3={caches['cache_l3']}"

    def test_detect_cpu_with_cache_size_fallback(self):
        """/proc/cpuinfo 的 cache size 作为 L3 降级。"""
        cpuinfo_no_sys = SAMPLE_CPUINFO.replace(
            "cache size\t: 122880 KB", "cache size\t: 122880 KB"
        )
        with patch.object(platform, "system", return_value="Linux"):
            with patch.object(platform, "machine", return_value="x86_64"):
                with patch("builtins.open", mock_open(read_data=cpuinfo_no_sys)):
                    with patch("os.path.exists", return_value=False):
                        with patch("os.listdir", return_value=[]):
                            with patch("os.path.isdir", return_value=False):
                                result = detect_cpu()

        # L3 should come from cache size field
        assert "122880" in result["cache_l3"]

    def test_detect_cpu_max_freq_from_sysfs(self):
        """最大频率从 /sys/.../cpufreq 获取。"""
        with patch.object(platform, "system", return_value="Linux"):
            with patch.object(platform, "machine", return_value="x86_64"):
                with patch("builtins.open") as mock_file:
                    def side_effect(path, *args, **kwargs):
                        if path == "/proc/cpuinfo":
                            return mock_open(read_data=SAMPLE_CPUINFO).return_value
                        elif "cpuinfo_max_freq" in path:
                            m = mock_open()
                            m.return_value.read.return_value = "3800000"
                            return m.return_value
                        raise FileNotFoundError(f"Unexpected path: {path}")

                    def exists_side_effect(path):
                        if "cpu0/cpufreq/cpuinfo_max_freq" in path:
                            return True
                        return False

                    mock_file.side_effect = side_effect

                    with patch("os.path.exists", side_effect=exists_side_effect):
                        with patch("os.listdir", return_value=[]):
                            with patch("os.path.isdir", return_value=False):
                                result = detect_cpu()

        assert result["max_frequency_mhz"] == 3800.0


# ============================================================
# TestDetectMemory
# ============================================================


class TestDetectMemory:
    def test_detect_memory_with_dmidecode(self):
        """dmidecode 提供完整内存信息。"""
        with patch.object(platform, "system", return_value="Linux"):
            with patch("builtins.open", mock_open(read_data=SAMPLE_MEMINFO)):
                with patch(
                    "reasoning_env_test.detectors.system.system_info._run_dmidecode",
                    return_value=SAMPLE_DMIDECODE,
                ):
                    result = detect_memory()

        assert result["total_gb"] > 0
        assert result["type"] == "DDR5"
        assert result["speed_mhz"] == 4800
        assert result["modules"] == 2
        assert result["size_per_module_gb"] == 96.0

    def test_detect_memory_fallback(self):
        """无 dmidecode 时只返回总量。"""
        with patch.object(platform, "system", return_value="Linux"):
            with patch("builtins.open", mock_open(read_data=SAMPLE_MEMINFO)):
                with patch(
                    "reasoning_env_test.detectors.system.system_info._run_dmidecode",
                    return_value=None,
                ):
                    result = detect_memory()

        assert result["total_gb"] > 0
        assert result["type"] == "Unknown"
        assert result["speed_mhz"] == 0
        assert result["modules"] == 0

    def test_detect_memory_no_meminfo(self):
        """/proc/meminfo 不存在时返回全零。"""
        with patch.object(platform, "system", return_value="Linux"):
            with patch("builtins.open", side_effect=FileNotFoundError):
                with patch(
                    "reasoning_env_test.detectors.system.system_info._run_dmidecode",
                    return_value=None,
                ):
                    result = detect_memory()

        assert result["total_gb"] == 0.0
        assert result["type"] == "Unknown"

    def test_detect_memory_non_linux(self):
        """非 Linux 系统返回默认值。"""
        with patch.object(platform, "system", return_value="Windows"):
            result = detect_memory()

        assert result["total_gb"] == 0.0

    def test_detect_memory_empty_dmidecode_sections(self):
        """dmidecode 输出无 Memory Device 时不崩溃。"""
        empty_dmi = "# dmidecode 3.5\nHandle 0x0039, DMI type 16, 23 bytes\nPhysical Memory Array\n"
        with patch.object(platform, "system", return_value="Linux"):
            with patch("builtins.open", mock_open(read_data=SAMPLE_MEMINFO)):
                with patch(
                    "reasoning_env_test.detectors.system.system_info._run_dmidecode",
                    return_value=empty_dmi,
                ):
                    result = detect_memory()

        assert result["type"] == "Unknown"
        assert result["modules"] == 0


# ============================================================
# TestDetectAll
# ============================================================


class TestDetectAll:
    def _mock_uname(self, kernel="5.15.0-91-generic"):
        """Helper: mock platform.uname() to return a namedtuple with .release."""
        from collections import namedtuple
        uname_info = namedtuple("uname_result", ["release"])
        return patch.object(platform, "uname", return_value=uname_info(release=kernel))

    def test_detect_all_returns_dict(self):
        """detect_all() 返回包含 os/cpu/memory 的字典。"""
        with patch.object(platform, "system", return_value="Linux"):
            with patch("builtins.open") as mock_file:
                def side_effect(path, *args, **kwargs):
                    if "os-release" in path:
                        return mock_open(read_data=SAMPLE_OS_RELEASE).return_value
                    if "cpuinfo" in path:
                        return mock_open(read_data=SAMPLE_CPUINFO).return_value
                    if "meminfo" in path:
                        return mock_open(read_data=SAMPLE_MEMINFO).return_value
                    raise FileNotFoundError(f"Unexpected: {path}")
                mock_file.side_effect = side_effect

                with patch.object(socket, "gethostname", return_value="server-01"):
                    with patch.object(platform, "machine", return_value="x86_64"):
                        with self._mock_uname("5.15.0-91-generic"):
                            with patch("os.path.exists", return_value=False):
                                with patch("os.listdir", return_value=[]):
                                    with patch(
                                        "reasoning_env_test.detectors.system.system_info._run_dmidecode",
                                        return_value=None,
                                    ):
                                        result = detect_all()

        assert isinstance(result, dict)
        assert "os" in result
        assert "cpu" in result
        assert "memory" in result
        assert result["os"]["distribution"] == "ubuntu"
        assert result["cpu"]["model_name"] == "Intel(R) Xeon(R) Platinum 8468"
        assert result["memory"]["total_gb"] > 0

    def test_detect_all_no_exceptions(self):
        """detect_all() 在所有文件缺失时不抛异常。"""
        with patch.object(platform, "system", return_value="Linux"):
            with patch("builtins.open", side_effect=FileNotFoundError):
                with patch.object(socket, "gethostname", side_effect=Exception):
                    with patch.object(platform, "machine", return_value=""):
                        with patch("os.path.exists", return_value=False):
                            result = detect_all()

        assert isinstance(result, dict)
        assert "os" in result
        assert "cpu" in result
        assert "memory" in result


# ============================================================
# _run_dmidecode
# ============================================================


class TestRunDmidecode:
    def test_dmidecode_not_found(self):
        """dmidecode 命令不存在。"""
        with patch(
            "reasoning_env_test.detectors.system.system_info.shutil_which",
            return_value=None,
        ):
            from reasoning_env_test.detectors.system.system_info import _run_dmidecode
            assert _run_dmidecode() is None

    def test_dmidecode_success(self):
        """dmidecode 成功执行。"""
        with patch(
            "reasoning_env_test.detectors.system.system_info.shutil_which",
            return_value="/usr/sbin/dmidecode",
        ):
            with patch.object(subprocess, "run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = SAMPLE_DMIDECODE
                mock_run.return_value = mock_result

                from reasoning_env_test.detectors.system.system_info import _run_dmidecode
                result = _run_dmidecode()

                assert result == SAMPLE_DMIDECODE

    def test_dmidecode_fails_then_sudo(self):
        """dmidecode 失败后尝试 sudo。"""
        with patch(
            "reasoning_env_test.detectors.system.system_info.shutil_which",
            return_value="/usr/sbin/dmidecode",
        ):
            with patch.object(subprocess, "run") as mock_run:
                def side_effect(cmd, *args, **kwargs):
                    if cmd == ["dmidecode", "-t", "memory"]:
                        mock_result = MagicMock()
                        mock_result.returncode = 1
                        mock_result.stdout = ""
                        return mock_result
                    elif cmd[:2] == ["sudo", "-n"]:
                        mock_result = MagicMock()
                        mock_result.returncode = 0
                        mock_result.stdout = SAMPLE_DMIDECODE
                        return mock_result
                    raise ValueError(f"Unexpected cmd: {cmd}")

                mock_run.side_effect = side_effect

                from reasoning_env_test.detectors.system.system_info import _run_dmidecode
                result = _run_dmidecode()

                assert result == SAMPLE_DMIDECODE
