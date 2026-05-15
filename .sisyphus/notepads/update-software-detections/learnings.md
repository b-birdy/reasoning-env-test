2026-05-15 - Task: Update software.py with more commands and hardware_tools

- Expanded COMMANDS list from 9 to 21 entries (added xpu-smi, ip, ethtool, rdma, ibstat, lspci, dmidecode, lscpu, lsblk, lshw, numactl, perf)
- Added HARDWARE_TOOLS constant (12 entries) with detect_hardware_tools() using shutil.which()
- Unified scan_bin_dirs() to always scan BIN_DIRS + PATH (removed platform branch, deduplicated)
- Added "all_commands" and "hardware_tools" fields to detect_all() output
- Updated tests: TestScanBinDirs (new unified logic), TestDetectHardwareTools (new), TestDetectAll (new fields)
- All 32 tests pass
