"""模型推荐器：根据硬件配置推荐可部署的模型。"""
import json
import os
from typing import Dict, List, Any, Union


class ModelRecommender:
    """根据检测到的显存清单，从预定义模型JSON匹配可部署模型，输出排序推荐列表。"""

    def __init__(self, hardware_results: List[Dict[str, Any]]):
        """
        初始化模型推荐器。

        Args:
            hardware_results: 硬件检测结果列表，每个元素是字典，包含 "type", "memory_total_gb" 等字段。
        """
        self.hardware_results = hardware_results
        self.model_list_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "model_list.json"
        )
        self.models = self._load_model_list()

    def _load_model_list(self) -> List[Dict[str, Any]]:
        """加载模型列表JSON文件。"""
        with open(self.model_list_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def recommend(self) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        生成模型推荐列表。

        Returns:
            推荐列表，每个元素包含 "model", "precision", "gpu_index" 字段；
            如果没有匹配模型，返回 {"error": True, "message": "...", "min_required": {...}}。
        """
        # 1. 按类型分组统计显存
        gpu_types = ["nvidia", "amd", "ascend", "kunlunxin", "hygon"]
        gpu_groups = {}
        total_gpu_memory = 0.0

        for hw in self.hardware_results:
            hw_type = hw["type"]
            if hw_type not in gpu_types or hw["memory_total_gb"] <= 0:
                continue

            # 支持多卡（xpu_count/gpu_count）
            details = hw.get("details", {})
            gpu_count = details.get("gpu_count", 0) or details.get("xpu_count", 0) or 1

            if hw_type not in gpu_groups:
                gpu_groups[hw_type] = []

            for _ in range(gpu_count):
                gpu_groups[hw_type].append(hw)
                total_gpu_memory += hw["memory_total_gb"]

        if not gpu_groups:
            return {
                "error": True,
                "message": "未检测到可用于模型部署的GPU硬件",
                "min_required": {"gpu_required": True}
            }

        # 2. 计算可用显存（支持单卡和多卡合并）
        available_options = []
        for gpu_type, gpu_list in gpu_groups.items():
            # 单卡选项
            for i, gpu in enumerate(gpu_list):
                available_options.append({
                    "memory": gpu["memory_total_gb"],
                    "type": gpu_type,
                    "gpu_index": i,
                    "is_multi": False
                })

            # 多卡合并选项（如果有多个同类型GPU）
            if len(gpu_list) > 1:
                total_memory = sum(g["memory_total_gb"] for g in gpu_list)
                available_options.append({
                    "memory": total_memory,
                    "type": gpu_type,
                    "gpu_index": 0,
                    "is_multi": True
                })

        # 3. 匹配模型
        recommendations = []
        all_requirements = []

        for model in self.models:
            # 检查不同精度的显存需求
            precisions = [
                ("fp16", model["min_vram_fp16_gb"]),
                ("int8", model["min_vram_int8_gb"]),
                ("int4", model["min_vram_int4_gb"])
            ]
            matched = False
            for option in available_options:
                for precision, required in precisions:
                    if option["memory"] >= required:
                        recommendations.append({
                            "model": model,
                            "precision": precision,
                            "gpu_index": option["gpu_index"],
                            "gpu_type": option["type"],
                            "is_multi_gpu": option["is_multi"]
                        })
                        matched = True
                        break
                if matched:
                    break
            # 记录最小需求用于错误信息
            min_req = min([req for _, req in precisions])
            all_requirements.append({
                "model": model["name"],
                "params_b": model["params_b"],
                "min_vram_fp16": model["min_vram_fp16_gb"],
                "min_vram_int8": model["min_vram_int8_gb"],
                "min_vram_int4": model["min_vram_int4_gb"]
            })

        if not recommendations:
            # 找到对显存要求最小的模型
            min_vram_model = min(all_requirements, key=lambda x: x["min_vram_int4"])
            return {
                "error": True,
                "message": f"显存不足以部署任何推荐模型。当前总可用显存: {total_gpu_memory:.1f}GB",
                "min_required": {
                    "total_gpu_memory_gb": total_gpu_memory,
                    "lightest_model": min_vram_model
                }
            }

        # 4. 按参数量从大到小排序，去重
        # 去重策略：同一模型只保留最高精度的推荐
        unique_recs = {}
        for rec in recommendations:
            model_name = rec["model"]["name"]
            if model_name not in unique_recs:
                unique_recs[model_name] = rec
            else:
                # 比较精度优先级：fp16 > int8 > int4
                existing_prec = unique_recs[model_name]["precision"]
                new_prec = rec["precision"]
                prec_order = {"fp16": 3, "int8": 2, "int4": 1}
                if prec_order.get(new_prec, 0) > prec_order.get(existing_prec, 0):
                    unique_recs[model_name] = rec

        # 最终排序：参数量从大到小
        sorted_recs = sorted(
            unique_recs.values(),
            key=lambda x: -x["model"]["params_b"]
        )

        return sorted_recs
