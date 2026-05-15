# 推理环境检测报告

**生成时间**: 2026-05-15 14:13:22

---

## 1. 概览

以下表格汇总服务器核心配置概要。

| 项目 | 内容 |
|---|---|
| 主机名 | DESKTOP-WZXPC |
| 操作系统 | Unknown Unknown |
| 内核版本 | 10 |
| CPU 型号 | Unknown |
| CPU 核心数 | 0 物理 / 0 逻辑 |
| 系统内存 | 31.8 GB |
| GPU/加速卡 | 1×NVIDIA GeForce RTX 4070 Ti SUPER (16 GB /卡) |
| 网络 | - |

---

## 2. 系统信息

### 2.1 操作系统

| 项目 | 内容 |
|---|---|
| 发行版 | Unknown |
| 版本 | Unknown |
| 内核 | 10 |
| 主机名 | DESKTOP-WZXPC |
| 架构 | AMD64 |

### 2.2 CPU 详情

| 项目 | 内容 |
|---|---|
| 型号 | Unknown |
| 架构 | AMD64 |
| 物理核心 | 0 |
| 逻辑核心 | 0 |
| CPU 插槽 | 0 |

### 2.3 内存详情

| 项目 | 内容 |
|---|---|
| 总量 | Unknown |
| 类型 | Unknown |

---

## 3. GPU/加速卡

### 3.1 Nvidia

- **型号**: NVIDIA GeForce RTX 4070 Ti SUPER
- **数量**: 1 张
- **总显存**: 16.0 GB（单卡）

> **GPU 汇总**: 共 **1** 张加速卡，总计 **16 GB** 显存。

---

## 4. 网络

网络检测未执行或无数据（非 Linux 系统或无 ip 命令）。

---

## 5. 软件环境

### 5.1 Python 环境

- **版本**: 3.11.9  ✅ 满足（要求 ≥ 3.8）

### 5.2 推理框架

| 框架 | 版本 | 状态 | 安装建议 |
|---|---|---|---|
| vllm | - | ⚠️ 未安装 | pip install vllm |
| text-generation | - | ⚠️ 未安装 | pip install text-generation |
| ollama | - | ⚠️ 未安装 | 参考 https://ollama.com/download |
| tensorrt-llm | - | ⚠️ 未安装 | pip install tensorrt_llm |
| onnxruntime | 1.24.3 | ✅ 已安装 | - |
| torch | 2.11.0 | ✅ 已安装 | - |
| tensorflow | - | ⚠️ 未安装 | pip install tensorflow |

### 5.3 CUDA / ROCm

- **CUDA**: ✅ 13.1
- **ROCm**: ⚠️ 未检测到（仅 AMD GPU 需要）

### 5.4 硬件管理工具

| 工具 | 用途 | 状态 |
|---|---|---|
| dmidecode | 硬件信息查询 | ⚠️ 未安装 |
| ethtool | 网卡信息查询 | ⚠️ 未安装 |
| ibstat | InfiniBand 状态 | ⚠️ 未安装 |
| ip | 网络接口管理 | ⚠️ 未安装 |
| lscpu | CPU 信息查询 | ⚠️ 未安装 |
| lshw | 硬件配置查询 | ⚠️ 未安装 |
| lspci | PCIe 设备列表 | ⚠️ 未安装 |
| npu-smi | 昇腾 NPU 监控 | ⚠️ 未安装 |
| nvidia-smi | NVIDIA GPU 监控 | ✅ 已安装 |
| rdma | RDMA 设备管理 | ⚠️ 未安装 |
| rocm-smi | AMD GPU 监控 | ⚠️ 未安装 |
| xpu-smi | 昆仑芯 XPU 监控 | ⚠️ 未安装 |

---

## 6. 可部署模型推荐

根据服务器显存配置，以下模型可在当前环境中部署运行（按推荐优先级排列）。

| 推荐模型 | 参数量 | 推荐精度 | 显存占用 | 模型类型 | 推理框架建议 | ModelScope |
|---|---|---|---|---|---|---|
| InternLM2.5-20B-Chat | 20.0B | int4 | 12 GB | LLM | vLLM + AWQ / GPTQ | [链接](https://modelscope.cn/models/Shanghai_AI_Laboratory/InternLM2.5-20B-Chat) |
| Qwen2.5-14B-Instruct | 14.0B | int4 | 9 GB | LLM | vLLM + AWQ / GPTQ | [链接](https://modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct) |
| Llama-3.2-11B-Vision-Instruct | 11.0B | int8 | 12 GB | 多模态 | vLLM + AWQ / GPTQ | [链接](https://modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct) |
| Llama-3.1-8B-Instruct | 8.0B | int8 | 9 GB | LLM | vLLM + AWQ / GPTQ | [链接](https://modelscope.cn/models/LLM-Research/Llama-3.1-8B-Instruct) |
| Qwen2.5-7B-Instruct | 7.0B | int8 | 8 GB | LLM | vLLM + AWQ / GPTQ | [链接](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct) |
| Qwen2.5-VL-7B-Instruct | 7.0B | int8 | 9 GB | 多模态 | vLLM + AWQ / GPTQ | [链接](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct) |
| InternLM2.5-7B-Chat | 7.0B | int8 | 8 GB | LLM | vLLM + AWQ / GPTQ | [链接](https://modelscope.cn/models/Shanghai_AI_Laboratory/InternLM2.5-7B-Chat) |
| Mistral-7B-Instruct-v0.3 | 7.0B | int8 | 8 GB | LLM | vLLM + AWQ / GPTQ | [链接](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-Instruct-v0.3) |
| Yi-1.5-6B-Chat | 6.0B | fp16 | 14 GB | LLM | vLLM / TGI | [链接](https://modelscope.cn/models/01-ai/Yi-1.5-6B-Chat) |
| Qwen2.5-3B-Instruct | 3.0B | fp16 | 7 GB | LLM | vLLM / TGI | [链接](https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct) |
| Llama-3.2-3B-Instruct | 3.0B | fp16 | 7 GB | LLM | vLLM / TGI | [链接](https://modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct) |
| Qwen2.5-1.5B-Instruct | 1.5B | fp16 | 4 GB | LLM | vLLM / TGI | [链接](https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct) |

---

## 7. 性能预估

以下为不同并发级别下的理论性能估算值（基于推荐模型）。

| 并发数 | 吞吐 (tok/s) | P50 延迟 (ms) | P99 延迟 (ms) | 最大支持并发 |
|---|---|---|---|---|
| 4 | 5292.0 | 1443.5 | 2887.1 | 16 |
| 8 | 5292.0 | 1777.2 | 3554.4 | 16 |
| 16 | 5292.0 | 2188.0 | 4376.0 | 16 |
| 32 | 5292.0 | 2693.7 | 5387.5 | 16 |
| 64 | 5292.0 | 3316.4 | 6632.8 | 16 |
| 128 | 5292.0 | 4082.9 | 8165.9 | 16 |

> ⚠️ **性能折损说明**：以上数据为理论估算值，基于 40% 的
> 理论到实际性能折损系数（即实际性能约为理论值的 60%）。
> **实际性能**可能因硬件型号、驱动版本、软件栈、工作负载特征、
> 批处理策略等而与估算值存在显著差异。
> 本数据**不可替代**真实环境下的性能基准测试。

---


*报告由推理环境检测工具自动生成*