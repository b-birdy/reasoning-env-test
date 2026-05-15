"""Container environment detection module.

Detects whether the code is running inside a Docker container,
Kubernetes pod, and checks the availability of Docker CLI as well
as cgroup resource limits.
"""

import os
import shutil


def detect_container_env() -> dict:
    """Detect if the current process runs inside a Docker container.

    Uses multiple indicators:
      - Presence of ``/.dockerenv`` file (classic Docker marker).
      - Presence of ``"docker"`` in ``/proc/1/cgroup``.
      - Non-standard PID 1 scheduler entry in ``/proc/1/sched``.

    Returns
    -------
    dict
        ``{"in_container": bool, "container_type": str | None}``
        where *container_type* is ``"docker"`` or ``None``.
    """
    in_container = False
    container_type = None

    # --- Check 1: /.dockerenv marker file ---
    if os.path.exists("/.dockerenv"):
        in_container = True
        container_type = "docker"

    # --- Check 2: /proc/1/cgroup contains "docker" ---
    if not in_container:
        try:
            with open("/proc/1/cgroup", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if "docker" in content:
                in_container = True
                container_type = "docker"
        except (FileNotFoundError, IOError, PermissionError):
            pass

    # --- Check 3: /proc/1/sched differs from typical host PID 1 ---
    # On a bare-metal Linux host PID 1 is usually "init" or "systemd";
    # inside a container it is the entrypoint (e.g. "python", "bash").
    if not in_container:
        try:
            with open("/proc/1/sched", encoding="utf-8", errors="replace") as f:
                first_line = f.readline().strip()
            # Typical host PID 1 names
            host_init_names = {"init", "systemd"}
            proc_name = first_line.split(" ")[0] if first_line else ""
            if proc_name and proc_name not in host_init_names:
                # This alone is a weak signal; mark only as hint
                pass
        except (FileNotFoundError, IOError, PermissionError):
            pass

    return {"in_container": in_container, "container_type": container_type}


def detect_k8s_env() -> bool:
    """Detect if running inside a Kubernetes Pod.

    Relies on the ``KUBERNETES_SERVICE_HOST`` environment variable,
    which is automatically injected by Kubernetes into every Pod.

    Returns
    -------
    bool
        ``True`` if the environment variable is set (non-empty).
    """
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST"))


def detect_docker_installed() -> bool:
    """Detect whether the Docker CLI is available on ``PATH``.

    Returns
    -------
    bool
        ``True`` if ``docker`` executable is found.
    """
    return shutil.which("docker") is not None


def detect_resource_limits() -> dict:
    """Detect the container's memory and CPU limits via cgroup v1.

    Reads the following control-group files:
      - ``/sys/fs/cgroup/memory/memory.limit_in_bytes``
      - ``/sys/fs/cgroup/cpu/cpu.cfs_quota_us``

    On non-Linux systems, or when the files are not present
    (e.g. cgroup v2), both values default to ``None``.

    Returns
    -------
    dict
        ``{"memory_limit_bytes": int | None, "cpu_quota": int | None}``
    """
    memory_limit = None
    cpu_quota = None

    try:
        with open(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            encoding="utf-8",
            errors="replace",
        ) as f:
            memory_limit = int(f.read().strip())
    except (FileNotFoundError, IOError, PermissionError, ValueError):
        pass

    try:
        with open(
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
            encoding="utf-8",
            errors="replace",
        ) as f:
            cpu_quota = int(f.read().strip())
    except (FileNotFoundError, IOError, PermissionError, ValueError):
        pass

    return {"memory_limit_bytes": memory_limit, "cpu_quota": cpu_quota}


def detect_all() -> dict:
    """Run all container-related checks and return a unified result.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``in_container`` (*bool*) – ``True`` if any container indicator
          matched.
        - ``container_type`` (*str* or ``None``) – ``"docker"``, ``"k8s"``,
          or ``None``.
        - ``k8s_pod`` (*bool*) – ``True`` when the process runs in a
          Kubernetes Pod.
        - ``docker_installed`` (*bool*) – ``True`` if the Docker CLI is
          on ``PATH``.
        - ``memory_limit_bytes`` (*int* or ``None``) – cgroup memory limit.
        - ``cpu_quota`` (*int* or ``None``) – cgroup CPU quota.
    """
    container_info = detect_container_env()
    k8s_pod = detect_k8s_env()
    docker_installed = detect_docker_installed()
    resource_limits = detect_resource_limits()

    # If K8s is detected, promote the container type
    if k8s_pod:
        container_info["in_container"] = True
        container_info["container_type"] = "k8s"

    return {
        "in_container": container_info["in_container"],
        "container_type": container_info["container_type"],
        "k8s_pod": k8s_pod,
        "docker_installed": docker_installed,
        "memory_limit_bytes": resource_limits["memory_limit_bytes"],
        "cpu_quota": resource_limits["cpu_quota"],
    }
