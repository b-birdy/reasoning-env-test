# 推理环境检测报告

**生成时间**: 2026-05-15 13:40:24
**服务器IP**: 10.180.34.31

---

## 1. 服务器配置检查清单

以下表格汇总了服务器的硬件配置、已安装软件、推理框架及容器环境状态。

| 检查项目 | 检测结果 | 状态 | 建议 |
|---|---|---|---|
| CPU 型号 | x86_64 | ✅ 正常 | - |
| CPU 架构 | x86_64 | ✅ 正常 | - |
| CPU 核心数 | 192 | ✅ 正常 | - |
| 系统内存 | 1511.5 GB | ✅ 正常 | - |
| GPU 硬件 | 未检测到 | ⚠️ 缺失 | 服务器无 GPU 或驱动未安装，无法部署 GPU 推理模型 |
| 命令: python3 | 已安装 | ✅ 正常 | - |
| 命令: nvidia-smi | 未安装 | ⚠️ 缺失 | 安装 NVIDIA 驱动: https://www.nvidia.com/drivers/ — NVIDIA GPU 监控与管理工具 |
| 命令: rocm-smi | 未安装 | ⚠️ 缺失 | 安装 ROCm: https://rocm.docs.amd.com/ — AMD GPU 监控与管理工具 |
| 命令: npu-smi | 未安装 | ⚠️ 缺失 | 安装昇腾 NPU 驱动: https://www.hiascend.com/ — 昇腾 NPU 监控与管理工具 |
| 命令: docker | 已安装 | ✅ 正常 | - |
| 命令: kubectl | 未安装 | ⚠️ 缺失 | 安装 kubectl: https://kubernetes.io/docs/tasks/tools/ — Kubernetes 集群管理，大规模部署必需 |
| 命令: ollama | 未安装 | ⚠️ 缺失 | 安装 Ollama: https://ollama.com/download — 本地大模型快速部署服务 |
| 命令: pip | 已安装 | ✅ 正常 | - |
| 框架: vllm | 未安装 | ⚠️ 缺失 | pip install vllm — 推理部署推荐安装 |
| 框架: text-generation | 未安装 | ⚠️ 缺失 | pip install text-generation — 推理部署推荐安装 |
| 框架: ollama | 未安装 | ⚠️ 缺失 | 参考 https://ollama.com/download — 推理部署推荐安装 |
| 框架: tensorrt-llm | 未安装 | ⚠️ 缺失 | pip install tensorrt_llm — 推理部署推荐安装 |
| 框架: onnxruntime | 未安装 | ⚠️ 缺失 | pip install onnxruntime — 推理部署推荐安装 |
| 框架: torch | 2.5.1+cu121 | ✅ 正常 | - |
| 框架: tensorflow | 未安装 | ⚠️ 缺失 | pip install tensorflow — 推理部署推荐安装 |
| Python 版本 | 3.10.12 | ✅ 正常 | ≥ 3.8，满足要求 |
| CUDA 版本 | 未检测到 | ⚠️ 缺失 | 安装 NVIDIA 驱动及 CUDA Toolkit 以启用 GPU 推理 |
| ROCm 版本 | 未检测到 | ℹ️ 不适用 | 仅 AMD GPU 需要 |
| 容器环境 | 否 (裸金属/虚拟机) | ✅ 正常 | - |
| Docker CLI | 已安装 | ✅ 正常 | - |

---

## 2. 可部署模型推荐

根据服务器显存配置，以下模型可在当前环境中部署运行（按参数量从大到小排序）。

### 模型推荐结果

⚠️ **未检测到可用于模型部署的GPU硬件**

> **最低配置建议**: 需要至少一张 GPU（NVIDIA/AMD/昇腾/昆仑芯/海光）
> 并安装对应驱动程序，方可部署推理模型。

---

## 3. 性能预估

以下为各并发级别下的理论性能估算值。

⚠️ 无法进行性能预估（缺少硬件或模型信息）。

---


*报告由推理环境检测工具自动生成*