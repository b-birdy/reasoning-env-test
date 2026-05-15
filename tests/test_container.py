"""Unit tests for the container environment detection module."""

import os
from unittest.mock import mock_open, patch

from reasoning_env_test.detectors.container import (
    detect_all,
    detect_container_env,
    detect_docker_installed,
    detect_k8s_env,
    detect_resource_limits,
)


class TestDetectContainerEnv:
    """Tests for :func:`detect_container_env`."""

    @patch("os.path.exists")
    @patch("builtins.open")
    def test_dockerenv_file_exists(self, mock_open_file, mock_exists):
        """Detect container when /.dockerenv is present."""
        # /.dockerenv exists → short-circuit to docker
        def exists_side_effect(path):
            return path == "/.dockerenv"

        mock_exists.side_effect = exists_side_effect

        result = detect_container_env()

        assert result["in_container"] is True
        assert result["container_type"] == "docker"
        # open should NOT be called because /.dockerenv short-circuits
        mock_open_file.assert_not_called()

    @patch("os.path.exists")
    @patch("builtins.open")
    def test_cgroup_contains_docker(self, mock_open_file, mock_exists):
        """Detect container when /proc/1/cgroup contains 'docker'."""
        mock_exists.return_value = False  # /.dockerenv not present

        # Simulate /proc/1/cgroup content with "docker"
        cgroup_content = (
            "12:hugetlb:/\n"
            "11:devices:/docker/abc123\n"
            "10:memory:/docker/abc123\n"
        )
        mock_open_file.return_value = mock_open(read_data=cgroup_content).return_value

        result = detect_container_env()

        assert result["in_container"] is True
        assert result["container_type"] == "docker"
        # Verify the correct file was opened
        mock_open_file.assert_called_once_with(
            "/proc/1/cgroup", encoding="utf-8", errors="replace"
        )

    @patch("os.path.exists")
    @patch("builtins.open")
    def test_not_in_container(self, mock_open_file, mock_exists):
        """No container indicators present."""
        mock_exists.return_value = False

        # cgroup doesn't contain docker
        cgroup_content = "12:hugetlb:/\n11:devices:/\n10:memory:/\n"
        mock_open_file.return_value = mock_open(read_data=cgroup_content).return_value

        result = detect_container_env()

        assert result["in_container"] is False
        assert result["container_type"] is None

    @patch("os.path.exists")
    @patch("builtins.open")
    def test_cgroup_file_not_found(self, mock_open_file, mock_exists):
        """No indicators and /proc/1/cgroup is absent (e.g. Windows)."""
        mock_exists.return_value = False
        mock_open_file.side_effect = FileNotFoundError(
            "/proc/1/cgroup does not exist"
        )

        result = detect_container_env()

        assert result["in_container"] is False
        assert result["container_type"] is None

    @patch("os.path.exists")
    @patch("builtins.open")
    def test_cgroup_permission_denied(self, mock_open_file, mock_exists):
        """Gracefully handle permission errors when reading cgroup."""
        mock_exists.return_value = False
        mock_open_file.side_effect = PermissionError("Permission denied")

        result = detect_container_env()

        assert result["in_container"] is False
        assert result["container_type"] is None


class TestDetectK8sEnv:
    """Tests for :func:`detect_k8s_env`."""

    @patch.dict(
        os.environ,
        {"KUBERNETES_SERVICE_HOST": "10.96.0.1", "KUBERNETES_SERVICE_PORT": "443"},
        clear=True,
    )
    def test_k8s_environment_variable_set(self):
        """Detect K8s when KUBERNETES_SERVICE_HOST is set."""
        assert detect_k8s_env() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_k8s_environment_variable_not_set(self):
        """No K8s detection when env variable is absent."""
        assert detect_k8s_env() is False

    @patch.dict(
        os.environ,
        {"KUBERNETES_SERVICE_HOST": ""},
        clear=True,
    )
    def test_k8s_environment_variable_empty(self):
        """Empty KUBERNETES_SERVICE_HOST should not count as detected."""
        assert detect_k8s_env() is False


class TestDetectDockerInstalled:
    """Tests for :func:`detect_docker_installed`."""

    @patch("shutil.which")
    def test_docker_installed(self, mock_which):
        """Docker CLI is on PATH."""
        mock_which.return_value = "/usr/bin/docker"
        assert detect_docker_installed() is True
        mock_which.assert_called_once_with("docker")

    @patch("shutil.which")
    def test_docker_not_installed(self, mock_which):
        """Docker CLI is not on PATH."""
        mock_which.return_value = None
        assert detect_docker_installed() is False
        mock_which.assert_called_once_with("docker")


class TestDetectResourceLimits:
    """Tests for :func:`detect_resource_limits`."""

    @patch("builtins.open")
    def test_linux_cgroup_files_present(self, mock_open_file):
        """Read memory and CPU limits on a Linux system with cgroup v1."""
        mock_open_file.side_effect = [
            mock_open(read_data="8388608\n").return_value,  # memory
            mock_open(read_data="50000\n").return_value,  # cpu
        ]

        result = detect_resource_limits()

        assert result["memory_limit_bytes"] == 8388608
        assert result["cpu_quota"] == 50000

    @patch("builtins.open")
    def test_cgroup_files_not_found(self, mock_open_file):
        """Cgroup files missing (non-Linux or cgroup v2)."""
        mock_open_file.side_effect = FileNotFoundError("File not found")

        result = detect_resource_limits()

        assert result["memory_limit_bytes"] is None
        assert result["cpu_quota"] is None

    @patch("builtins.open")
    def test_cgroup_read_error(self, mock_open_file):
        """IO error while reading cgroup files."""
        mock_open_file.side_effect = IOError("Read error")

        result = detect_resource_limits()

        assert result["memory_limit_bytes"] is None
        assert result["cpu_quota"] is None

    @patch("builtins.open")
    def test_cgroup_value_error(self, mock_open_file):
        """Non-integer content in cgroup files."""
        mock_open_file.side_effect = [
            mock_open(read_data="invalid\n").return_value,
            mock_open(read_data="invalid\n").return_value,
        ]

        result = detect_resource_limits()

        assert result["memory_limit_bytes"] is None
        assert result["cpu_quota"] is None

    @patch("builtins.open")
    def test_partial_cgroup_data(self, mock_open_file):
        """Memory file readable, CPU file missing."""
        mock_open_file.side_effect = [
            mock_open(read_data="16777216\n").return_value,
            FileNotFoundError("CPU file not found"),
        ]

        result = detect_resource_limits()

        assert result["memory_limit_bytes"] == 16777216
        assert result["cpu_quota"] is None

    @patch("builtins.open")
    def test_no_limit_value(self, mock_open_file):
        """Memory limit is a very large number (9223372036854771712 = no limit)."""
        mock_open_file.side_effect = [
            mock_open(read_data="9223372036854771712\n").return_value,
            mock_open(read_data="-1\n").return_value,
        ]

        result = detect_resource_limits()

        # We report the raw value; caller decides how to interpret it
        assert result["memory_limit_bytes"] == 9223372036854771712
        assert result["cpu_quota"] == -1


class TestDetectAll:
    """Tests for :func:`detect_all` (unified entry point)."""

    @patch("shutil.which")
    @patch("builtins.open")
    @patch("os.path.exists")
    @patch.dict(os.environ, {}, clear=True)
    def test_full_docker_scenario(
        self, mock_exists, mock_open_file, mock_which
    ):
        """Unified result when running in Docker with resource limits."""
        # /.dockerenv exists → Docker container
        mock_exists.side_effect = lambda p: p == "/.dockerenv"

        # cgroup files provide limits
        mock_open_file.side_effect = [
            mock_open(read_data="4194304\n").return_value,  # memory
            mock_open(read_data="100000\n").return_value,  # cpu
        ]

        mock_which.return_value = "/usr/bin/docker"

        result = detect_all()

        assert result == {
            "in_container": True,
            "container_type": "docker",
            "k8s_pod": False,
            "docker_installed": True,
            "memory_limit_bytes": 4194304,
            "cpu_quota": 100000,
        }

    @patch("shutil.which")
    @patch("builtins.open")
    @patch("os.path.exists")
    @patch.dict(
        os.environ,
        {"KUBERNETES_SERVICE_HOST": "10.96.0.1"},
        clear=True,
    )
    def test_full_k8s_scenario(
        self, mock_exists, mock_open_file, mock_which
    ):
        """Unified result when running in a K8s Pod."""
        mock_exists.return_value = False  # no /.dockerenv

        # cgroup does NOT contain "docker" (k8s uses different paths)
        cgroup_content = (
            "12:hugetlb:/\n"
            "11:devices:/kubepods/burstable/pod123/abc\n"
            "10:memory:/kubepods/burstable/pod123/abc\n"
        )
        # Calls: detect_container_env (/proc/1/cgroup, /proc/1/sched)
        #       + detect_resource_limits (memory, cpu) = 4 total
        mock_open_file.side_effect = [
            mock_open(read_data=cgroup_content).return_value,  # /proc/1/cgroup
            mock_open(read_data="bash (1, #threads: 1)\n").return_value,  # /proc/1/sched
            mock_open(read_data="8388608\n").return_value,  # memory limit
            mock_open(read_data="-1\n").return_value,  # cpu quota
        ]

        mock_which.return_value = None

        result = detect_all()

        assert result == {
            "in_container": True,
            "container_type": "k8s",
            "k8s_pod": True,
            "docker_installed": False,
            "memory_limit_bytes": 8388608,
            "cpu_quota": -1,
        }

    @patch("shutil.which")
    @patch("builtins.open")
    @patch("os.path.exists")
    @patch.dict(os.environ, {}, clear=True)
    def test_no_container_scenario(
        self, mock_exists, mock_open_file, mock_which
    ):
        """Unified result when not inside any container."""
        mock_exists.return_value = False

        cgroup_content = "12:hugetlb:/\n11:devices:/\n10:memory:/\n"
        # Calls: detect_container_env (cgroup, sched) + detect_resource_limits (memory, cpu)
        mock_open_file.side_effect = [
            mock_open(read_data=cgroup_content).return_value,  # /proc/1/cgroup
            mock_open(read_data="systemd (1, #threads: 1)\n").return_value,  # /proc/1/sched
            FileNotFoundError("No cgroup memory file"),
            FileNotFoundError("No cgroup cpu file"),
        ]

        mock_which.return_value = None

        result = detect_all()

        assert result == {
            "in_container": False,
            "container_type": None,
            "k8s_pod": False,
            "docker_installed": False,
            "memory_limit_bytes": None,
            "cpu_quota": None,
        }

    @patch("shutil.which")
    @patch("builtins.open")
    @patch("os.path.exists")
    @patch.dict(os.environ, {}, clear=True)
    def test_docker_installed_no_container(
        self, mock_exists, mock_open_file, mock_which
    ):
        """Docker CLI installed but not running inside a container."""
        mock_exists.return_value = False
        cgroup_content = "12:hugetlb:/\n11:devices:/\n10:memory:/\n"
        # Calls: detect_container_env (cgroup, sched) + detect_resource_limits (memory, cpu)
        mock_open_file.side_effect = [
            mock_open(read_data=cgroup_content).return_value,  # /proc/1/cgroup
            mock_open(read_data="systemd (1, #threads: 1)\n").return_value,  # /proc/1/sched
            FileNotFoundError("No cgroup memory file"),
            FileNotFoundError("No cgroup cpu file"),
        ]
        mock_which.return_value = "/usr/bin/docker"

        result = detect_all()

        assert result == {
            "in_container": False,
            "container_type": None,
            "k8s_pod": False,
            "docker_installed": True,
            "memory_limit_bytes": None,
            "cpu_quota": None,
        }
