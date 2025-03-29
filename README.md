# QwenSuiCoder

A project to fine-tune Qwen2.5-Coder models for Sui Move smart contract development and SDK usage using PyTorch and SLoRA.

## Project Overview

QwenSuiCoder aims to create a specialized LLM capable of:
- Writing and debugging Sui Move smart contracts
- Working with Sui SDKs in TypeScript and Python
- Following best practices for the Sui blockchain ecosystem

## Features

- **Qwen2.5-Coder Models**: Uses the latest state-of-the-art Qwen2.5-Coder models (0.5B to 32B parameters)
- **Pure PyTorch Implementation**: No dependencies on Hugging Face
- **SLoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with Sparse Low-Rank Adaptation
- **Hardware-aware Benchmarking**: Automatically finds optimal model size and settings for your hardware
- **Unified Management**: Single interface for benchmarking, training, and inference
- **Ollama Integration**: Automatically configures the best model in Ollama for easy deployment

## Project Structure

```
qwensuicoder/
├── benchmark_qwen_pytorch.py   # Benchmark script for finding the best Qwen2.5-Coder model
├── train_qwen_pytorch.py       # Training script with SLoRA implementation
├── model_manager.py            # Unified interface for PyTorch, Ollama, and SLora
├── run_qwen_pytorch.sh         # Interactive script for the entire pipeline
├── clear_cuda_cache.py         # Utility to maximize available GPU memory
├── config/                     # Configuration files for model training
├── data/                       # Training data directory
├── examples/                   # Example Sui Move code and SDK usage
├── benchmark_results/          # Benchmark results and optimal settings
├── trained_models/             # Directory for saving trained models
└── ollama_config/              # Ollama configuration files
```

## Setup

### Prerequisites

- Python 3.9+
- PyTorch with CUDA support
- Ollama installed and in PATH
- GPU with sufficient VRAM (at least 8GB recommended)

### Environment Setup

```bash
# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x run_qwen_pytorch.sh model_manager.py
```

## The Complete Pipeline

We've created a unified pipeline to:
1. Benchmark your hardware to find the best Qwen2.5-Coder model size
2. Fine-tune the model with SLoRA for your specific use case
3. Configure Ollama for easy model deployment and inference

### Interactive Script

The easiest way to get started is with our interactive script:

```bash
./run_qwen_pytorch.sh
```

This script provides a menu-driven interface with the following options:
1. Run benchmark to find optimal model
2. Train model with SLora
3. Configure Ollama with best model
4. Run inference with Ollama
5. Run inference with PyTorch
6. Run full pipeline (benchmark → train → configure)

### Model Manager

For programmatic control, use the model manager:

```bash
# Run benchmark
python model_manager.py --action benchmark --max-model-size 14

# Train model with SLora
python model_manager.py --action train --epochs 3 --learning-rate 2e-4

# Configure Ollama with best model
python model_manager.py --action configure-ollama

# Run inference with Ollama
python model_manager.py --action infer-ollama --prompt "Write a Sui Move smart contract for a counter."

# Run inference with PyTorch directly
python model_manager.py --action infer-pytorch --prompt "Write a Sui Move smart contract for a counter."
```

## Qwen2.5-Coder Models

This project supports the following Qwen2.5-Coder models:

| Model | Parameters | Min VRAM | Description |
|-------|------------|----------|-------------|
| Qwen2.5-Coder-0.5B | 0.5B | 2GB | Smallest model, good for limited hardware |
| Qwen2.5-Coder-1.5B | 1.5B | 4GB | Good balance for consumer GPUs |
| Qwen2.5-Coder-7B | 7B | 8GB | Recommended minimum for serious code generation |
| Qwen2.5-Coder-14B | 14B | 16GB | Excellent performance/size tradeoff |
| Qwen2.5-Coder-32B | 32B | 32GB | State-of-the-art performance, requires high-end GPU |

The benchmark will automatically find the best model for your hardware and configure it for both PyTorch and Ollama.

## SLoRA: Sparse Low-Rank Adaptation

SLoRA extends LoRA by adding sparsity to the low-rank matrices, further improving parameter efficiency:

- **Memory efficiency**: Requires up to 95% less memory than full fine-tuning
- **Training speed**: Trains significantly faster than full fine-tuning
- **Size efficiency**: SLoRA weights are typically <10MB regardless of model size
- **Task adaptability**: Performs exceptionally well on domain-specific tasks

## License

MIT 