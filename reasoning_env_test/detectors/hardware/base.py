from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseHardwareDetector(ABC):
    """硬件检测器抽象基类。

    所有硬件检测器（CPU、GPU、NPU 等）必须继承此类并实现 detect() 方法。
    检测结果遵循统一输出格式。
    """

    @abstractmethod
    def detect(self) -> Dict[str, Any]:
        """检测硬件信息并返回统一格式的字典。

        Returns:
            统一格式字典，包含以下字段：
                - type (str): 硬件类型标识，如 "cpu", "nvidia", "amd" 等
                - model (str): 硬件型号名称
                - memory_total_gb (float): 总内存/显存大小（GB）
                - compute_units (int): 计算单元数量（CPU核心数、GPU SM数量等）
                - details (dict): 该硬件类型的额外详细信息
        """

    def _empty_result(self, hw_type: str) -> Dict[str, Any]:
        """返回空检测结果（硬件不存在时使用）。

        Args:
            hw_type: 硬件类型标识

        Returns:
            所有字段为默认值的统一格式字典
        """
        return {
            "type": hw_type,
            "model": "",
            "memory_total_gb": 0.0,
            "compute_units": 0,
            "details": {},
        }
