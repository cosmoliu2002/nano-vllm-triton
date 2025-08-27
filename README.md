# Nano-vLLM-Triton

[English](README.md) | [中文](README_zh.md)

A lightweight and optimized vLLM implementation built on [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) and [OpenAI Triton](https://github.com/openai/triton).

## 🆕 What's New

This project extends the original Nano-vLLM with the following improvements:

1. **Extended Model Support**: Added comprehensive support for:
   - Qwen2 series models
   - Qwen2.5 series models
   - Qwen3-MoE series models
   - Llama series models
   - Original Qwen3 series models

2. **Triton Optimization**: Implemented custom Triton operators:
   - `softmax_online`: Replaces `torch.softmax`
   - `cat_cos_sin`: Replaces `torch.cat(cos, sin, dim=-1)`
   - **Performance Gain**: Average speed improvement of 101.93 tokens/s

## 📦 Installation

```bash
git clone https://github.com/cosmoliu2002/nano-vllm-triton.git
cd nano-vllm-triton
pip install -e .
```

## 🚀 Quick Start

### 1. Python

**example.py**
```bash
python example.py --model-path /path/to/your/model
```

**Supported Parameters**
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
**Supported Parameters**
**`config_example.yaml`**
```yaml
model_path: "/path/to/your/model"
tensor_parallel_size: 1
enforce_eager: true
temperature: 0.6
max_tokens: 256
```

## 📊 Performance Benchmark

**run_bench.sh**

```bash
bash run_bench.sh
```
**Supported Parameters**
**`config_bench.yaml`**
```yaml
model_path: "/path/to/your/model"
engine: "nanovllmtriton"             # nanovllmtriton or vllm
```

**Test Configuration:**
- Hardware: NVIDIA A10 (Powered by ModelScope Notebook)
- Models: Various Qwen and Llama models
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens
- Test Runs: 5 runs per model with average results

**Test Results:**

| Model | vLLM Throughput (tokens/s) | Nano vLLM Triton Throughput (tokens/s) | Nano vLLM Throughput (tokens/s) | Performance Gain (vs Nano vLLM) |
|-------|---------------------------|----------------------------------------|----------------------------------|----------------------------------|
| Llama3.2-1B | 5455.85 | 5432.24 | 5394.15 | +38.09 |
| Qwen2-0.5B | 9501.11 | 10198.84 | 10030.08 | +168.76 |
| Qwen2.5-0.5B | 9441.46 | 10221.94 | 10033.20 | +188.74 |
| Qwen3-0.6B | 3465.40 | 2970.81 | 2958.67 | +12.14 |

## 🛠️ Supported Models

- **Qwen Series**: Qwen2, Qwen2.5, Qwen3, Qwen3-MoE
- **Llama Series**: Llama 2, Llama 3, Llama 3.1, Llama 3.2

## 🗺️ Future Roadmap

Gradually replace all PyTorch implementations with custom Triton kernels

## 🙏 Acknowledgments

- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
- Qwen2, Qwen2.5, Qwen3-MoE, Llama series model support [Nano-vLLM-gogongxt](https://github.com/gogongxt/nano-vllm)
- [OpenAI Triton](https://github.com/openai/triton)
