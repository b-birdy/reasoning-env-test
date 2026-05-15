"""Container environment detection package."""

from .container import (
    detect_container_env,
    detect_k8s_env,
    detect_docker_installed,
    detect_resource_limits,
    detect_all,
)

__all__ = [
    "detect_container_env",
    "detect_k8s_env",
    "detect_docker_installed",
    "detect_resource_limits",
    "detect_all",
]
