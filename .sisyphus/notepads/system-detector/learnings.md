# System Detector Learnings

## Files Created
- `reasoning_env_test/detectors/system/__init__.py` - Module exports
- `reasoning_env_test/detectors/system/system_info.py` - OS/CPU/Memory detection
- `tests/test_system.py` - Unit tests (20 tests)

## Implementation Patterns
- **OS detection**: `/etc/os-release` for distro+version, `platform.uname().release` for kernel, `socket.gethostname()`, `platform.machine()` for arch
- **CPU detection**: `/proc/cpuinfo` parsing (most reliable), differentiate physical vs logical cores via `physical id` + `core id` + `cpu cores` fields
- **Cache detection**: Priority order: (1) `/proc/cpuinfo` l1d/l1i/l2/l3 cache fields, (2) `/sys/.../cache/index{idx}/{type,size,level}`, (3) traditional `cache size` field (=L3)
- **Memory detection**: `/proc/meminfo` for total (always available), `dmidecode -t memory` for type/speed/modules (optional, needs root)
- All functions wrapped in try/except, no exceptions propagate

## Gotchas (Windows-specific)
- `mock.patch.object(platform.uname, "release")` fails because `platform.uname` is a **function**, not a module. Use `patch.object(platform, "uname", return_value=namedtuple(...))` instead
- `os.path.join` on Windows produces backslashes (`\`), so test data with forward slashes (`/`) won't match mocked `open()` calls. Normalize with `path.replace(os.sep, "/")`
- `platform.system()` should be wrapped in try/except too (test might mock it to raise)

## Cache Level Detection (sysfs)
Each `/sys/devices/system/cpu/cpu0/cache/index{idx}/` has:
- `type`: Data, Instruction, or Unified
- `size`: e.g. "48K", "2M", "120M"  
- `level`: 1, 2, or 3

Mapping logic:
- Level 1 + Data → cache_l1d
- Level 1 + Instruction → cache_l1i
- Level 1 + Unified → cache_l1d (fallback)
- Level 2 → cache_l2
- Level 3 → cache_l3
