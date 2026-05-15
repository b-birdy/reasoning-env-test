# Reasoning Env Test

AI 大语言推理环境检测工具。

## 简介

Reasoning Env Test 是一个命令行工具，用于检测当前机器的硬件和软件环境是否满足主流 AI 大语言模型的推理需求。它能够自动识别各类加速硬件、检测驱动和框架版本，并给出模型推荐与性能预估。

## 功能

- **硬件检测**：检测 CPU、NVIDIA GPU、AMD GPU、昇腾 NPU、昆仑芯 XPU、海光 DCU 等硬件设备
- **软件检测**：检测操作系统、Python 版本、CUDA / ROCm / CANN 等加速框架
- **容器环境检测**：识别 Docker / Podman 等容器运行时环境
- **模型推荐**：根据检测结果推荐适合本地运行的大语言模型
- **性能预估**：估算硬件对常见模型的推理性能
- **报告输出**：生成可读的环境检测报告

## 系统要求

- Python 3.8 或更高版本
- 支持 Windows、Linux、macOS

## 安装

### 方式一：从源码安装

```bash
git clone https://github.com/your-username/reasoning-env-test.git
cd reasoning-env-test
pip install -r requirements.txt
```

> 当前项目暂未提供 requirements.txt，依赖项定义在 pyproject.toml 中。你也可以直接使用 pip 从本地安装：

### 方式二：pip 本地安装

```bash
git clone https://github.com/your-username/reasoning-env-test.git
cd reasoning-env-test
pip install .
```

安装开发依赖（运行测试需要）：

```bash
pip install -e ".[dev]"
```

## 使用方法

### 查看帮助

```bash
python -m reasoning_env_test --help
```

### 运行检测

```bash
python -m reasoning_env_test
```

直接运行将自动检测当前机器的硬件和软件环境，并输出检测报告。

### 参数说明

| 参数 | 说明 |
|------|------|
| `--help` | 显示帮助信息 |
| `--output` / `-o` | 指定报告输出路径（支持 JSON / 文本格式） |
| `--verbose` / `-v` | 输出详细日志信息 |
| `--no-color` | 禁用彩色输出 |

> 具体参数以 `python -m reasoning_env_test --help` 输出为准。

### 运行测试

```bash
pip install -e ".[dev]"
pytest
```

## 示例输出

运行检测后，工具会输出类似以下内容的环境信息：

```
========================================
  Reasoning Env Test Report
========================================
系统信息
  OS: Windows 10.0.19045
  Python: 3.10.11
  CPU: Intel(R) Core(TM) i7-10700K

硬件加速器
  NVIDIA GPU: GeForce RTX 3090 (24GB)
  CUDA 版本: 12.1

框架检测
  PyTorch: 2.1.0 (CUDA 可用)
  TensorFlow: 2.13.0

推荐模型
  - Qwen2.5-7B-Q4 (7GB VRAM)
  - LLaMA-3-8B-Q4 (8GB VRAM)

性能预估
  GeForce RTX 3090: 约 40-50 tokens/s (7B Q4)
========================================
```

> 实际输出内容取决于当前机器的硬件配置。

## 支持硬件

| 硬件类型 | 检测模块 |
|----------|----------|
| NVIDIA GPU | `detectors/hardware/nvidia.py` |
| AMD GPU | `detectors/hardware/amd.py` |
| 昇腾 NPU (Ascend) | `detectors/hardware/ascend.py` |
| 昆仑芯 XPU (Kunlunxin) | `detectors/hardware/kunlunxin.py` |
| 海光 DCU (Hygon) | `detectors/hardware/hygon.py` |
| CPU | `detectors/hardware/cpu.py` |

## 目录结构

```
reasoning-env-test/
├── reasoning_env_test/        # 主包
│   ├── __init__.py
│   ├── data/                  # 模型数据
│   │   ├── __init__.py
│   │   └── model_list.json    # 模型参数列表
│   ├── detectors/             # 检测器
│   │   ├── container/         # 容器环境检测
│   │   │   ├── __init__.py
│   │   │   └── container.py
│   │   ├── hardware/          # 硬件检测
│   │   │   ├── __init__.py
│   │   │   ├── base.py        # 硬件检测基类
│   │   │   ├── cpu.py
│   │   │   ├── nvidia.py
│   │   │   ├── amd.py
│   │   │   ├── ascend.py
│   │   │   ├── kunlunxin.py
│   │   │   └── hygon.py
│   │   └── software/          # 软件检测
│   │       ├── __init__.py
│   │       └── software.py
│   ├── performance/           # 性能预估
│   │   └── __init__.py
│   ├── recommend/             # 模型推荐
│   │   └── __init__.py
│   └── report/                # 报告输出
│       └── __init__.py
├── tests/                     # 测试
│   ├── __init__.py
│   ├── test_container.py
│   ├── test_hardware.py
│   └── test_software.py
├── pyproject.toml              # 项目配置与依赖
└── README.md
```

## 许可证

本项目基于 MIT 许可证开源。详见项目根目录的 LICENSE 文件。
