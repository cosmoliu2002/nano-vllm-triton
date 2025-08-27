# Nano-vLLM-Triton

[English](README.md) | [中文](README_zh.md)

基于 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 和 [OpenAI Triton](https://github.com/openai/triton) 构建的轻量级、优化版 vLLM 实现。

## 🆕 更新内容

本项目在原始 Nano-vLLM 基础上进行了以下改进：

1. **扩展模型支持**：新增全面支持：
   - Qwen2 系列模型
   - Qwen2.5 系列模型
   - Qwen3-MoE 系列模型
   - Llama 系列模型
   - 原有的 Qwen3 系列模型

2. **Triton 优化**：实现自定义 Triton 算子：
   - `softmax_online`：替换 `torch.softmax`
   - `cat_cos_sin`：替换 `torch.cat(cos, sin, dim=-1)`
   - **性能提升**：平均速度提升 101.93 token/s

## 📦 安装

```bash
git clone https://github.com/cosmoliu2002/nano-vllm-triton.git
cd nano-vllm-triton
pip install -e .
```

## 🚀 快速开始

### 1. Python

**example.py**
```bash
python example.py --model-path /path/to/your/model
```

**支持参数**
```bash
--model-path
--tensor-parallel-size
--enforce-eager
--temperature
--max-tokens
```

### 2. Shell

**run_example.sh**

```bash
bash run_example.sh
```
**支持参数**
**`config_example.yaml`**
```yaml
model_path: "/path/to/your/model"
tensor_parallel_size: 1
enforce_eager: true
temperature: 0.6
max_tokens: 256
```

## 📊 性能测试

**run_bench.sh**

```bash
bash run_bench.sh
```
**支持参数**
**`config_bench.yaml`**
```yaml
model_path: "/path/to/your/model"
engine: "nanovllmtriton"             # nanovllmtriton 或 vllm
```

**测试配置：**
- 硬件：NVIDIA A10（由ModelScope Notebook提供支持）
- 模型：多种 Qwen 和 Llama 模型
- 总请求数：256 个序列
- 输入长度：100-1024 tokens 随机采样
- 输出长度：100-1024 tokens 随机采样
- 测试次数：每个模型测试5次取平均值

**测试结果：**

| 模型 | vLLM 吞吐量 (tokens/s) | Nano vLLM Triton 吞吐量 (tokens/s) | Nano vLLM 吞吐量 (tokens/s) | 性能提升(vs Nano vLLM) |
|-------|---------------------------|----------------------------------------|----------------------------------|----------------------------------|
| Llama3.2-1B | 5455.85 | 5432.24 | 5394.15 | +38.09 |
| Qwen2-0.5B | 9501.11 | 10198.84 | 10030.08 | +168.76 |
| Qwen2.5-0.5B | 9441.46 | 10221.94 | 10033.20 | +188.74 |
| Qwen3-0.6B | 3465.40 | 2970.81 | 2958.67 | +12.14 |

## 🛠️ 支持的模型

- **Qwen 系列**：Qwen2、Qwen2.5、Qwen3、Qwen3-MoE
- **Llama 系列**：Llama 2、Llama 3、Llama 3.1、Llama 3.2

## 🗺️ 未来规划

通过逐步使用自定义 Triton 内核替换所有 PyTorch 实现

## 🙏 致谢

- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
- Qwen2、Qwen2.5、Qwen3-MoE、Llama 系列模型支持 [Nano-vLLM-gogongxt](https://github.com/gogongxt/nano-vllm)
- [OpenAI Triton](https://github.com/openai/triton)