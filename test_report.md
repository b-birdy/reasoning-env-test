# 推理环境检测报告

**生成时间**: 2026-05-15 13:44:19

---

## 1. 服务器配置检查清单

以下表格汇总了服务器的硬件配置、已安装软件、推理框架及容器环境状态。

| 检查项目 | 检测结果 | 状态 | 建议 |
|---|---|---|---|
| CPU 型号 | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel | ✅ 正常 | - |
| CPU 架构 | AMD64 | ✅ 正常 | - |
| CPU 核心数 | 20 | ✅ 正常 | - |
| 系统内存 | 31.8 GB | ✅ 正常 | - |
| Nvidia GPU 型号 | NVIDIA GeForce RTX 4070 Ti SUPER | ✅ 正常 | 单卡 |
| Nvidia GPU 显存 | 16.0 GB | ✅ 正常 | - |
| 命令: python3 | 已安装 | ✅ 正常 | - |
| 命令: nvidia-smi | 已安装 | ✅ 正常 | - |
| 命令: rocm-smi | 未安装 | ⚠️ 缺失 | 安装 ROCm: https://rocm.docs.amd.com/ — AMD GPU 监控与管理工具 |
| 命令: npu-smi | 未安装 | ⚠️ 缺失 | 安装昇腾 NPU 驱动: https://www.hiascend.com/ — 昇腾 NPU 监控与管理工具 |
| 命令: docker | 已安装 | ✅ 正常 | - |
| 命令: kubectl | 已安装 | ✅ 正常 | - |
| 命令: ollama | 未安装 | ⚠️ 缺失 | 安装 Ollama: https://ollama.com/download — 本地大模型快速部署服务 |
| 命令: pip | 已安装 | ✅ 正常 | - |
| 框架: vllm | 未安装 | ⚠️ 缺失 | pip install vllm — 推理部署推荐安装 |
| 框架: text-generation | 未安装 | ⚠️ 缺失 | pip install text-generation — 推理部署推荐安装 |
| 框架: ollama | 未安装 | ⚠️ 缺失 | 参考 https://ollama.com/download — 推理部署推荐安装 |
| 框架: tensorrt-llm | 未安装 | ⚠️ 缺失 | pip install tensorrt_llm — 推理部署推荐安装 |
| 框架: onnxruntime | 1.24.3 | ✅ 正常 | - |
| 框架: torch | 2.11.0 | ✅ 正常 | - |
| 框架: tensorflow | 未安装 | ⚠️ 缺失 | pip install tensorflow — 推理部署推荐安装 |
| Python 版本 | 3.11.9 | ✅ 正常 | ≥ 3.8，满足要求 |
| CUDA 版本 | 13.1 | ✅ 正常 | 兼容 NVIDIA GPU 推理 |
| ROCm 版本 | 未检测到 | ℹ️ 不适用 | 仅 AMD GPU 需要 |
| 容器环境 | 否 (裸金属/虚拟机) | ✅ 正常 | - |
| Docker CLI | 已安装 | ✅ 正常 | - |

---

## 2. 可部署模型推荐

根据服务器显存配置，以下模型可在当前环境中部署运行（按参数量从大到小排序）。

| 推荐模型 | 参数量 | 推荐精度 | 显存占用 | 模型类型 | ModelScope链接 |
|---|---|---|---|---|---|
| InternLM2.5-20B-Chat | 20.0B | int4 | 12 GB | LLM | [链接](https://modelscope.cn/models/Shanghai_AI_Laboratory/InternLM2.5-20B-Chat) |
| Qwen2.5-14B-Instruct | 14.0B | int4 | 9 GB | LLM | [链接](https://modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct) |
| Llama-3.2-11B-Vision-Instruct | 11.0B | int8 | 12 GB | 多模态 | [链接](https://modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct) |
| Llama-3.1-8B-Instruct | 8.0B | int8 | 9 GB | LLM | [链接](https://modelscope.cn/models/LLM-Research/Llama-3.1-8B-Instruct) |
| Qwen2.5-7B-Instruct | 7.0B | int8 | 8 GB | LLM | [链接](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct) |
| Qwen2.5-VL-7B-Instruct | 7.0B | int8 | 9 GB | 多模态 | [链接](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct) |
| InternLM2.5-7B-Chat | 7.0B | int8 | 8 GB | LLM | [链接](https://modelscope.cn/models/Shanghai_AI_Laboratory/InternLM2.5-7B-Chat) |
| Mistral-7B-Instruct-v0.3 | 7.0B | int8 | 8 GB | LLM | [链接](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-Instruct-v0.3) |
| Yi-1.5-6B-Chat | 6.0B | fp16 | 14 GB | LLM | [链接](https://modelscope.cn/models/01-ai/Yi-1.5-6B-Chat) |
| Qwen2.5-3B-Instruct | 3.0B | fp16 | 7 GB | LLM | [链接](https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct) |
| Llama-3.2-3B-Instruct | 3.0B | fp16 | 7 GB | LLM | [链接](https://modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct) |
| Qwen2.5-1.5B-Instruct | 1.5B | fp16 | 4 GB | LLM | [链接](https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct) |

---

## 3. 性能预估

以下为各并发级别下的理论性能估算值。

| 并发数 | 吞吐(tok/s) | P50延迟(ms) | P99延迟(ms) | 最大支持并发 |
|---|---|---|---|---|
| 4 | 5292.0 | 1443.5 | 2887.1 | 16 |
| 8 | 5292.0 | 1777.2 | 3554.4 | 16 |
| 16 | 5292.0 | 2188.0 | 4376.0 | 16 |
| 32 | 5292.0 | 2693.7 | 5387.5 | 16 |
| 64 | 5292.0 | 3316.4 | 6632.8 | 16 |
| 128 | 5292.0 | 4082.9 | 8165.9 | 16 |

> ⚠️ **性能折损说明**：以上性能数据为理论估算值，基于 40% 的
> 理论到实际性能折损系数（即实际性能约为理论值的 60%）。
> **实际性能**可能因硬件型号、驱动版本、软件栈、工作负载特征、
> 批处理策略等而与估算值存在显著差异。
> 本数据**不可替代**真实环境下的性能基准测试。

---


*报告由推理环境检测工具自动生成*