# QwenSuiCoder

A lightweight project to fine-tune Qwen2.5-1.5B-Instruct model for Sui Move smart contract development and SDK usage using PyTorch and SLoRA.

## Project Overview

QwenSuiCoder aims to create a specialized LLM capable of:
- Writing and debugging Sui Move smart contracts
- Working with Sui SDKs in TypeScript and Python
- Following best practices for the Sui blockchain ecosystem

## Features

- **Optimized for Qwen2.5-1.5B-Instruct**: Focuses on the efficient 1.5B parameter model that balances performance and resource requirements
- **Pure PyTorch Implementation**: No dependencies on Hugging Face
- **SLoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with Sparse Low-Rank Adaptation
- **Hardware-aware Benchmarking**: Automatically optimizes training for resource-constrained environments
- **Unified Management**: Single interface for benchmarking, training, and inference
- **Ollama Integration**: Automatically configures the model in Ollama for easy deployment

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
- GPU with at least 4GB VRAM (for 1.5B model)

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
1. Benchmark your hardware to optimize the 1.5B model configuration
2. Fine-tune the model with SLoRA for your specific use case
3. Configure Ollama for easy model deployment and inference

### Interactive Script

The easiest way to get started is with our interactive script:

```bash
./run_qwen_pytorch.sh
```

This script provides a menu-driven interface with the following options:
1. Run benchmark to optimize model settings
2. Train model with SLora
3. Configure Ollama with the model
4. Run inference with Ollama
5. Run inference with PyTorch
6. Run full pipeline (benchmark → train → configure)

### Model Manager

For programmatic control, use the model manager:

```bash
# Run benchmark
python model_manager.py --action benchmark --max-model-size 1.5

# Train model with SLora
python model_manager.py --action train --epochs 3 --learning-rate 2e-4

# Configure Ollama with optimized model
python model_manager.py --action configure-ollama

# Run inference with Ollama
python model_manager.py --action infer-ollama --prompt "Write a Sui Move smart contract for a counter."

# Run inference with PyTorch directly
python model_manager.py --action infer-pytorch --prompt "Write a Sui Move smart contract for a counter."
```

## Qwen2.5 Models

This project primarily uses the Qwen2.5-1.5B-Instruct model, which offers an excellent balance of performance and efficiency:

| Model | Parameters | Min VRAM | Description |
|-------|------------|----------|-------------|
| Qwen2.5-0.5B-Instruct | 0.5B | 2GB | Smallest model, good for extremely limited hardware |
| Qwen2.5-1.5B-Instruct | 1.5B | 4GB | **Recommended** - Excellent balance for consumer GPUs |
| Qwen2.5-7B-Instruct | 7B | 8GB | Larger model if you have more resources available |
| Qwen2.5-14B-Instruct | 14B | 16GB | High-performance model requiring high-end GPU |
| Qwen2.5-32B-Instruct | 32B | 32GB | State-of-the-art performance, requires server-grade GPU |

The 1.5B model is the focus of this project as it provides good performance while being accessible to developers with consumer-grade GPUs.

## SLoRA: Sparse Low-Rank Adaptation

SLoRA extends LoRA by adding sparsity to the low-rank matrices, further improving parameter efficiency:

- **Memory efficiency**: Requires up to 95% less memory than full fine-tuning
- **Training speed**: Trains significantly faster than full fine-tuning
- **Size efficiency**: SLoRA weights are typically <10MB regardless of model size
- **Task adaptability**: Performs exceptionally well on domain-specific tasks

## License

MIT 