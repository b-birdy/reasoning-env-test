"""网络检测器单元测试。"""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reasoning_env_test.detectors.network.network import (
    detect_all,
    detect_interfaces,
    _get_rdma_info,
    _get_speed_from_ethtool,
    _get_speed_from_sysfs,
    _get_iface_type_from_sysfs,
    _determine_iface_type,
    _parse_ip_link_output,
    _speed_to_gbps,
    _classify_speed,
    _empty_result,
    _is_rdma_by_name,
    _run_cmd,
    _read_sysfs,
)


# ============================================================
# _run_cmd / _read_sysfs
# ============================================================

class TestRunCmd:
    def test_run_cmd_success(self):
        """命令成功执行返回 stdout。"""
        with patch("reasoning_env_test.detectors.network.network.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "link/ether 52:54:00:12:34:56\n"
            mock_run.return_value = mock_result

            result = _run_cmd(["ip", "link", "show"])
            assert result == "link/ether 52:54:00:12:34:56\n"

    def test_run_cmd_nonzero_returncode(self):
        """命令返回非零退出码时返回 None。"""
        with patch("reasoning_env_test.detectors.network.network.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_run.return_value = mock_result

            result = _run_cmd(["ip", "link", "show"])
            assert result is None

    def test_run_cmd_file_not_found(self):
        """命令不存在时返回 None。"""
        with patch("reasoning_env_test.detectors.network.network.subprocess.run", side_effect=FileNotFoundError):
            result = _run_cmd(["nonexistent_cmd"])
            assert result is None

    def test_run_cmd_timeout(self):
        """命令超时时返回 None。"""
        with patch("reasoning_env_test.detectors.network.network.subprocess.run", side_effect=OSError):
            result = _run_cmd(["ip", "link", "show"])
            assert result is None


class TestReadSysfs:
    def test_read_sysfs_success(self):
        """读取存在的 sysfs 文件返回内容。"""
        with patch("builtins.open", MagicMock()) as mock_file:
            mock_file.return_value.__enter__.return_value.read.return_value = "400000\n"
            with patch("os.path.isfile", return_value=True):
                result = _read_sysfs("/sys/class/net/eth0/speed")
                assert result == "400000"

    def test_read_sysfs_not_found(self):
        """文件不存在时返回 None。"""
        with patch("os.path.isfile", return_value=False):
            result = _read_sysfs("/sys/class/net/eth0/speed")
            assert result is None

    def test_read_sysfs_permission_denied(self):
        """权限不足时返回 None。"""
        with patch("os.path.isfile", return_value=True):
            with patch("builtins.open", side_effect=PermissionError):
                result = _read_sysfs("/sys/class/net/eth0/speed")
                assert result is None


# ============================================================
# _empty_result
# ============================================================

class TestEmptyResult:
    def test_empty_result_structure(self):
        """空结果包含所有必需字段。"""
        result = _empty_result()
        assert result["interfaces"] == []
        assert result["rdma_devices"] == []
        assert result["total_network_ports"] == 0
        assert result["summary"]["400g_ports"] == 0
        assert result["summary"]["200g_ports"] == 0
        assert result["summary"]["25g_ports"] == 0
        assert result["summary"]["total_bandwidth_gbps"] == 0


# ============================================================
# _speed_to_gbps
# ============================================================

class TestSpeedToGbps:
    def test_none_returns_zero(self):
        assert _speed_to_gbps(None) == 0

    def test_int_mbps_divide(self):
        assert _speed_to_gbps(400000) == 400

    def test_sysfs_mbps_string(self):
        assert _speed_to_gbps("400000") == 400

    def test_ethtool_mbps(self):
        assert _speed_to_gbps("400000Mb/s") == 400
        assert _speed_to_gbps("25000Mb/s") == 25
        assert _speed_to_gbps("1000Mb/s") == 1

    def test_ethtool_gbps(self):
        assert _speed_to_gbps("100Gb/s") == 100
        assert _speed_to_gbps("40Gb/s") == 40

    def test_ethtool_tbps(self):
        assert _speed_to_gbps("1Tb/s") == 1000

    def test_invalid_string(self):
        assert _speed_to_gbps("unknown") == 0
        assert _speed_to_gbps("") == 0

    def test_ethtool_lowercase(self):
        assert _speed_to_gbps("400000mb/s") == 400
        assert _speed_to_gbps("100gb/s") == 100


# ============================================================
# _classify_speed
# ============================================================

class TestClassifySpeed:
    def test_400g(self):
        assert _classify_speed(400) == "400g"
        assert _classify_speed(800) == "400g"

    def test_200g(self):
        assert _classify_speed(200) == "200g"
        assert _classify_speed(399) == "200g"

    def test_100g(self):
        assert _classify_speed(100) == "100g"
        assert _classify_speed(199) == "100g"

    def test_50g(self):
        assert _classify_speed(50) == "50g"
        assert _classify_speed(99) == "50g"

    def test_25g(self):
        assert _classify_speed(25) == "25g"
        assert _classify_speed(49) == "25g"

    def test_10g(self):
        assert _classify_speed(10) == "10g"

    def test_other(self):
        assert _classify_speed(1) == "other"
        assert _classify_speed(0) == "other"


# ============================================================
# _is_rdma_by_name
# ============================================================

class TestIsRdmaByName:
    def test_ib_prefix(self):
        assert _is_rdma_by_name("ib0") is True
        assert _is_rdma_by_name("ib1") is True

    def test_rxe_pattern(self):
        assert _is_rdma_by_name("rxe_0") is True
        assert _is_rdma_by_name("rxe0") is True

    def test_mlx_pattern(self):
        assert _is_rdma_by_name("mlx5_0") is True
        assert _is_rdma_by_name("mlx5_1") is True

    def test_roce_pattern(self):
        assert _is_rdma_by_name("roce_0") is True

    def test_ethernet_not_rdma(self):
        assert _is_rdma_by_name("eth0") is False
        assert _is_rdma_by_name("ens1np0") is False
        assert _is_rdma_by_name("lo") is False

    def test_case_insensitive(self):
        assert _is_rdma_by_name("IB0") is True
        assert _is_rdma_by_name("MLX5_0") is True


# ============================================================
# _parse_ip_link_output
# ============================================================

SAMPLE_IP_LINK_OUTPUT = """1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000
    link/ether 52:54:00:12:34:56 brd ff:ff:ff:ff:ff:ff
3: ib0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 4092 qdisc mq state UP mode DEFAULT group default qlen 256
    link/infiniband 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00 brd 00:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff:ff
4: ens1np0: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 00:0a:f7:8b:6c:01 brd ff:ff:ff:ff:ff:ff
5: roce_0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000
    link/ether 00:02:c9:12:34:56 brd ff:ff:ff:ff:ff:ff
"""

SAMPLE_IP_LINK_WITH_VETH = """1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000
    link/ether 52:54:00:12:34:56 brd ff:ff:ff:ff:ff:ff
3: veth0@if2: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP mode DEFAULT group default qlen 1000
    link/ether 0a:1b:2c:3d:4e:5f brd ff:ff:ff:ff:ff:ff
"""


class TestParseIpLinkOutput:
    def test_parse_ethernet_interface(self):
        """正确解析以太网接口。"""
        results = _parse_ip_link_output(SAMPLE_IP_LINK_OUTPUT)
        eth0 = next(r for r in results if r["name"] == "eth0")
        assert eth0["index"] == 2
        assert eth0["state"] == "up"
        assert eth0["mac"] == "52:54:00:12:34:56"
        assert eth0["link_type"] == "ether"

    def test_parse_infiniband_interface(self):
        """正确解析 InfiniBand 接口。"""
        results = _parse_ip_link_output(SAMPLE_IP_LINK_OUTPUT)
        ib0 = next(r for r in results if r["name"] == "ib0")
        assert ib0["index"] == 3
        assert ib0["state"] == "up"
        assert ib0["link_type"] == "infiniband"
        # InfiniBand 地址是 20 字节 GUID，不应被当作 MAC 提取
        assert ib0["mac"] == ""

    def test_parse_loopback_interface(self):
        """正确解析 loopback 接口。"""
        results = _parse_ip_link_output(SAMPLE_IP_LINK_OUTPUT)
        lo = next(r for r in results if r["name"] == "lo")
        assert lo["index"] == 1
        assert lo["state"] == "unknown"
        assert lo["link_type"] == "loopback"
        assert lo["mac"] == "00:00:00:00:00:00"

    def test_parse_down_interface(self):
        """正确解析 DOWN 状态接口。"""
        results = _parse_ip_link_output(SAMPLE_IP_LINK_OUTPUT)
        down = next(r for r in results if r["name"] == "ens1np0")
        assert down["state"] == "down"
        assert down["mac"] == "00:0a:f7:8b:6c:01"

    def test_parse_roce_interface(self):
        """正确解析 RoCE 命名接口。"""
        results = _parse_ip_link_output(SAMPLE_IP_LINK_OUTPUT)
        roce = next(r for r in results if r["name"] == "roce_0")
        assert roce["index"] == 5
        assert roce["state"] == "up"
        assert roce["mac"] == "00:02:c9:12:34:56"

    def test_parse_veth_alias(self):
        """正确解析带 @ 别名的 veth 接口。"""
        results = _parse_ip_link_output(SAMPLE_IP_LINK_WITH_VETH)
        veth = next(r for r in results if r["name"] == "veth0")
        assert veth["index"] == 3
        assert veth["state"] == "up"
        assert veth["link_type"] == "ether"

    def test_parse_empty_output(self):
        """空输出返回空列表。"""
        results = _parse_ip_link_output("")
        assert results == []

    def test_parse_no_interfaces(self):
        """无接口输出返回空列表。"""
        results = _parse_ip_link_output("    \n\n")
        assert results == []

    def test_all_interfaces_have_required_fields(self):
        """每个解析后的接口都有必需字段。"""
        results = _parse_ip_link_output(SAMPLE_IP_LINK_OUTPUT)
        for iface in results:
            assert "index" in iface
            assert "name" in iface
            assert "state" in iface
            assert "link_type" in iface
            assert isinstance(iface["name"], str)
            assert isinstance(iface["state"], str)


# ============================================================
# _determine_iface_type
# ============================================================

class TestDetermineIfaceType:
    def test_infiniband_link_type(self):
        """链路类型为 infiniband 时返回 infiniband。"""
        assert _determine_iface_type("ib0", "infiniband", None, set()) == "infiniband"

    def test_loopback_link_type(self):
        """链路类型为 loopback 时返回 loopback。"""
        assert _determine_iface_type("lo", "loopback", None, set()) == "loopback"

    def test_ethernet_link_type_default(self):
        """链路类型为 ether 时按后续逻辑判定。"""
        result = _determine_iface_type("eth0", "ether", None, set())
        assert result == "ethernet"

    def test_sysfs_infiniband(self):
        """sysfs 类型为 infiniband 时返回 infiniband。"""
        result = _determine_iface_type("ib0", "ether", "infiniband", set())
        assert result == "infiniband"

    def test_rdma_netdev_roce(self):
        """在 rdma_netdevs 集合中的返回 roce。"""
        result = _determine_iface_type("eth0", "ether", None, {"eth0"})
        assert result == "roce"

    def test_name_based_infiniband(self):
        """名字含 ib 的返回 infiniband。"""
        result = _determine_iface_type("ib0", "ether", None, set())
        assert result == "infiniband"

    def test_name_based_roce_mlx(self):
        """名字含 mlx 的返回 roce。"""
        result = _determine_iface_type("mlx5_0", "ether", None, set())
        assert result == "roce"

    def test_default_ethernet(self):
        """无法判定时默认返回 ethernet。"""
        result = _determine_iface_type("eth0", "ether", None, set())
        assert result == "ethernet"


# ============================================================
# _get_rdma_info
# ============================================================

class TestGetRdmaInfo:
    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    def test_rdma_link_show_with_netdev(self, mock_run_cmd):
        """解析含 netdev 字段的 rdma link show 输出。"""
        mock_run_cmd.return_value = (
            "1: rxe_0: state ACTIVE physical_state LINK_UP netdev eth0\n"
            "2: mlx5_0: state ACTIVE physical_state LINK_UP netdev eth1\n"
        )
        devices, netdevs = _get_rdma_info()
        assert "rxe_0" in devices
        assert "mlx5_0" in devices
        assert "eth0" in netdevs
        assert "eth1" in netdevs

    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    def test_rdma_link_show_simple(self, mock_run_cmd):
        """解析简单格式的 rdma link show 输出。"""
        mock_run_cmd.return_value = (
            "1: rxe_0: state ACTIVE physical_state LINK_UP\n"
            "2: mlx5_0: state ACTIVE physical_state LINK_UP\n"
        )
        devices, netdevs = _get_rdma_info()
        assert "rxe_0" in devices
        assert "mlx5_0" in devices
        assert netdevs == set()

    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    def test_no_rdma_command_fallback_sysfs(self, mock_run_cmd):
        """rdma 命令不可用时回退到 sysfs。"""
        mock_run_cmd.return_value = None
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["mlx5_0", "mlx5_1"]):
                devices, netdevs = _get_rdma_info()
                assert "mlx5_0" in devices
                assert "mlx5_1" in devices
                assert netdevs == set()

    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    def test_no_rdma_devices(self, mock_run_cmd):
        """无 RDMA 设备时返回空列表。"""
        mock_run_cmd.return_value = None
        with patch("os.path.isdir", return_value=False):
            devices, netdevs = _get_rdma_info()
            assert devices == []
            assert netdevs == set()


# ============================================================
# _get_speed_from_ethtool
# ============================================================

class TestGetSpeedFromEthtool:
    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    def test_ethtool_speed(self, mock_run_cmd):
        """解析 ethtool 输出中的 Speed 字段。"""
        mock_run_cmd.return_value = (
            "Settings for eth0:\n"
            "    Speed: 400000Mb/s\n"
        )
        speed = _get_speed_from_ethtool("eth0")
        assert speed == 400

    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    def test_ethtool_speed_gbps(self, mock_run_cmd):
        """解析 ethtool 输出中 Gb/s 格式的速度。"""
        mock_run_cmd.return_value = (
            "Settings for eth0:\n"
            "    Speed: 100Gb/s\n"
        )
        speed = _get_speed_from_ethtool("eth0")
        assert speed == 100

    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    def test_ethtool_failure(self, mock_run_cmd):
        """ethtool 失败时返回 0。"""
        mock_run_cmd.return_value = None
        speed = _get_speed_from_ethtool("eth0")
        assert speed == 0

    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    def test_ethtool_no_speed_field(self, mock_run_cmd):
        """ethtool 输出中不含 Speed 字段时返回 0。"""
        mock_run_cmd.return_value = (
            "Settings for eth0:\n"
            "    No data available\n"
        )
        speed = _get_speed_from_ethtool("eth0")
        assert speed == 0


# ============================================================
# _get_speed_from_sysfs
# ============================================================

class TestGetSpeedFromSysfs:
    def test_speed_from_sysfs(self):
        """从 sysfs 读取速度。"""
        with patch("reasoning_env_test.detectors.network.network._read_sysfs", return_value="400000"):
            speed = _get_speed_from_sysfs("eth0")
            assert speed == 400

    def test_speed_not_available(self):
        """sysfs 无速度信息时返回 0。"""
        with patch("reasoning_env_test.detectors.network.network._read_sysfs", return_value=None):
            speed = _get_speed_from_sysfs("eth0")
            assert speed == 0

    def test_invalid_speed_value(self):
        """sysfs 返回 -1 时返回 0。"""
        with patch("reasoning_env_test.detectors.network.network._read_sysfs", return_value="-1"):
            speed = _get_speed_from_sysfs("eth0")
            assert speed == 0


# ============================================================
# _get_iface_type_from_sysfs
# ============================================================

class TestGetIfaceTypeFromSysfs:
    def test_ethernet_type(self):
        """sysfs 接口类型 1 对应 ethernet。"""
        with patch("reasoning_env_test.detectors.network.network._read_sysfs", return_value="1"):
            assert _get_iface_type_from_sysfs("eth0") == "ethernet"

    def test_infiniband_type(self):
        """sysfs 接口类型 32 对应 infiniband。"""
        with patch("reasoning_env_test.detectors.network.network._read_sysfs", return_value="32"):
            assert _get_iface_type_from_sysfs("ib0") == "infiniband"

    def test_loopback_type(self):
        """sysfs 接口类型 772 对应 loopback。"""
        with patch("reasoning_env_test.detectors.network.network._read_sysfs", return_value="772"):
            assert _get_iface_type_from_sysfs("lo") == "loopback"

    def test_unknown_type(self):
        """sysfs 未知类型号返回 None。"""
        with patch("reasoning_env_test.detectors.network.network._read_sysfs", return_value="999"):
            assert _get_iface_type_from_sysfs("eth0") is None

    def test_sysfs_unavailable(self):
        """sysfs 不可用时返回 None。"""
        with patch("reasoning_env_test.detectors.network.network._read_sysfs", return_value=None):
            assert _get_iface_type_from_sysfs("eth0") is None


# ============================================================
# detect_interfaces (集成测试 via mock)
# ============================================================

class TestDetectInterfaces:
    @patch("reasoning_env_test.detectors.network.network._get_rdma_info")
    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    @patch("reasoning_env_test.detectors.network.network._get_iface_type_from_sysfs")
    @patch("reasoning_env_test.detectors.network.network._get_speed_from_ethtool")
    @patch("reasoning_env_test.detectors.network.network._get_speed_from_sysfs")
    def test_basic_ethernet_and_infiniband(
        self,
        mock_sysfs_speed,
        mock_ethtool_speed,
        mock_sysfs_type,
        mock_run_cmd,
        mock_rdma,
    ):
        """检测以太网和 InfiniBand 接口。"""
        mock_rdma.return_value = (["mlx5_0"], {"eth0"})
        mock_run_cmd.return_value = SAMPLE_IP_LINK_OUTPUT
        mock_sysfs_type.return_value = None
        mock_ethtool_speed.return_value = 25
        mock_sysfs_speed.return_value = 0

        interfaces = detect_interfaces()
        # lo 应被跳过
        assert len(interfaces) == 4

        eth0 = next(i for i in interfaces if i["name"] == "eth0")
        # eth0 在 rdma_netdevs 中 → 类型应为 roce
        assert eth0["type"] == "roce"
        assert eth0["speed_gbps"] == 25
        assert eth0["state"] == "up"
        assert eth0.get("rdma") is True
        assert eth0["mac"] == "52:54:00:12:34:56"

        ib0 = next(i for i in interfaces if i["name"] == "ib0")
        assert ib0["type"] == "infiniband"
        assert ib0.get("rdma") is True
        # InfiniBand 接口无 MAC（非 ether 链路类型不提取）
        assert "mac" not in ib0

    @patch("reasoning_env_test.detectors.network.network._get_rdma_info")
    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    def test_ip_command_failure(self, mock_run_cmd, mock_rdma):
        """ip 命令失败时返回空列表。"""
        mock_rdma.return_value = ([], set())
        mock_run_cmd.return_value = None

        interfaces = detect_interfaces()
        assert interfaces == []

    @patch("reasoning_env_test.detectors.network.network._get_rdma_info")
    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    @patch("reasoning_env_test.detectors.network.network._get_iface_type_from_sysfs")
    @patch("reasoning_env_test.detectors.network.network._get_speed_from_ethtool")
    @patch("reasoning_env_test.detectors.network.network._get_speed_from_sysfs")
    def test_roce_interface_named(
        self,
        mock_sysfs_speed,
        mock_ethtool_speed,
        mock_sysfs_type,
        mock_run_cmd,
        mock_rdma,
    ):
        """名字含 roce 的接口被识别为 roce 类型。"""
        mock_rdma.return_value = ([], set())
        mock_run_cmd.return_value = SAMPLE_IP_LINK_OUTPUT
        mock_sysfs_type.return_value = None
        mock_ethtool_speed.return_value = 400
        mock_sysfs_speed.return_value = 0

        interfaces = detect_interfaces()
        roce = next(i for i in interfaces if i["name"] == "roce_0")
        assert roce["type"] == "roce"
        assert roce["speed_gbps"] == 400
        assert roce.get("rdma") is True

    @patch("reasoning_env_test.detectors.network.network._get_rdma_info")
    @patch("reasoning_env_test.detectors.network.network._run_cmd")
    @patch("reasoning_env_test.detectors.network.network._get_iface_type_from_sysfs")
    @patch("reasoning_env_test.detectors.network.network._get_speed_from_ethtool")
    @patch("reasoning_env_test.detectors.network.network._get_speed_from_sysfs")
    def test_fallback_to_sysfs_speed(
        self,
        mock_sysfs_speed,
        mock_ethtool_speed,
        mock_sysfs_type,
        mock_run_cmd,
        mock_rdma,
    ):
        """ethtool 失败时回退到 sysfs 获取速率。"""
        mock_rdma.return_value = ([], set())
        mock_run_cmd.return_value = SAMPLE_IP_LINK_OUTPUT
        mock_sysfs_type.return_value = None
        mock_ethtool_speed.return_value = 0  # ethtool 无结果
        mock_sysfs_speed.return_value = 100  # sysfs 回退

        interfaces = detect_interfaces()
        eth0 = next(i for i in interfaces if i["name"] == "eth0")
        assert eth0["speed_gbps"] == 100


# ============================================================
# detect_all
# ============================================================

class TestDetectAll:
    def test_non_linux_returns_empty(self):
        """非 Linux 系统返回空结果。"""
        with patch("reasoning_env_test.detectors.network.network.sys.platform", "win32"):
            result = detect_all()
            assert result["total_network_ports"] == 0
            assert result["interfaces"] == []
            assert result["summary"]["400g_ports"] == 0

    def test_no_ip_command_returns_empty(self):
        """无 ip 命令时返回空结果。"""
        with patch("reasoning_env_test.detectors.network.network.sys.platform", "linux"):
            with patch("reasoning_env_test.detectors.network.network.shutil.which", return_value=None):
                result = detect_all()
                assert result["total_network_ports"] == 0
                assert result["interfaces"] == []

    @patch("reasoning_env_test.detectors.network.network.sys.platform", "linux")
    @patch("reasoning_env_test.detectors.network.network.shutil.which", return_value="/usr/sbin/ip")
    @patch("reasoning_env_test.detectors.network.network.detect_interfaces")
    @patch("reasoning_env_test.detectors.network.network._get_rdma_info")
    def test_detect_all_success(
        self, mock_rdma_info, mock_detect, mock_which,
    ):
        """成功检测所有网络信息。"""
        mock_detect.return_value = [
            {"name": "eth0", "type": "ethernet", "speed_gbps": 25,
             "state": "up", "mac": "52:54:00:12:34:56"},
            {"name": "ens2np0", "type": "roce", "speed_gbps": 400,
             "state": "up", "rdma": True},
            {"name": "ens2np1", "type": "roce", "speed_gbps": 400,
             "state": "up", "rdma": True},
        ]
        mock_rdma_info.return_value = (["mlx5_0", "mlx5_1"], {"ens2np0", "ens2np1"})

        result = detect_all()
        assert result["total_network_ports"] == 3
        assert result["rdma_devices"] == ["mlx5_0", "mlx5_1"]
        assert result["summary"]["25g_ports"] == 1
        assert result["summary"]["400g_ports"] == 2
        assert result["summary"]["total_bandwidth_gbps"] == 825  # 25 + 400 + 400
        assert len(result["interfaces"]) == 3

    @patch("reasoning_env_test.detectors.network.network.sys.platform", "linux")
    @patch("reasoning_env_test.detectors.network.network.shutil.which", return_value="/usr/sbin/ip")
    @patch("reasoning_env_test.detectors.network.network.detect_interfaces")
    @patch("reasoning_env_test.detectors.network.network._get_rdma_info")
    def test_summary_with_no_speed(
        self, mock_rdma_info, mock_detect, mock_which,
    ):
        """无速率信息的接口不计入速率分类。"""
        mock_detect.return_value = [
            {"name": "ib0", "type": "infiniband", "speed_gbps": 0,
             "state": "up", "rdma": True},
        ]
        mock_rdma_info.return_value = ([], set())

        result = detect_all()
        assert result["total_network_ports"] == 1
        assert result["summary"]["400g_ports"] == 0
        assert result["summary"]["200g_ports"] == 0
        assert result["summary"]["25g_ports"] == 0
        assert result["summary"]["total_bandwidth_gbps"] == 0

    @patch("reasoning_env_test.detectors.network.network.sys.platform", "linux")
    @patch("reasoning_env_test.detectors.network.network.shutil.which", return_value="/usr/sbin/ip")
    @patch("reasoning_env_test.detectors.network.network.detect_interfaces")
    @patch("reasoning_env_test.detectors.network.network._get_rdma_info")
    def test_detect_all_server_scenario(
        self, mock_rdma_info, mock_detect, mock_which,
    ):
        """模拟真实服务器场景：8x400G RoCE + 2x200G RoCE + 3x25G ethernet。"""
        interfaces = []
        # 8 个 400G RoCE 口
        for i in range(8):
            interfaces.append({
                "name": f"ens{2+i}np0",
                "type": "roce",
                "speed_gbps": 400,
                "state": "up",
                "rdma": True,
            })
        # 2 个 200G RoCE 口
        for i in range(2):
            interfaces.append({
                "name": f"ens{10+i}np0",
                "type": "roce",
                "speed_gbps": 200,
                "state": "up",
                "rdma": True,
            })
        # 3 个 25G 以太网口
        for i in range(3):
            interfaces.append({
                "name": f"eth{i}",
                "type": "ethernet",
                "speed_gbps": 25,
                "state": "up",
            })

        mock_detect.return_value = interfaces
        mock_rdma_info.return_value = ([f"mlx5_{i}" for i in range(10)], set())

        result = detect_all()
        assert result["total_network_ports"] == 13
        assert result["summary"]["400g_ports"] == 8
        assert result["summary"]["200g_ports"] == 2
        assert result["summary"]["25g_ports"] == 3
        assert result["summary"]["total_bandwidth_gbps"] == 400 * 8 + 200 * 2 + 25 * 3
    