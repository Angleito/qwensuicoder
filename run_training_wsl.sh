#!/bin/bash

# WSL-specific wrapper for running the optimized training script
# This script disables DeepSpeed's CUDA extension builds since WSL may not have NVCC

echo "Setting up WSL-compatible environment for training..."

# Disable DeepSpeed CUDA extensions to avoid NVCC requirement
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_UTILS=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_TRANSFORMER_INFERENCE=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0
export DS_BUILD_SPARSE_ATTN=0
export TORCH_CUDA_ARCH_LIST="8.6"

# Check if CUDA is available through PyTorch
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
  echo "CUDA is available to PyTorch"
  
  # Try WSL-specific CUDA paths
  if [ -d "/usr/lib/wsl/lib" ]; then
    echo "Using WSL-specific CUDA path"
    export CUDA_HOME="/usr/lib/wsl/lib"
  else
    # Try to find where PyTorch's CUDA libraries are
    PYTORCH_CUDA_PATH=$(python -c "import torch; print(torch._C._cuda_getDirFor('nvrtc'))" 2>/dev/null || echo "")
    if [ ! -z "$PYTORCH_CUDA_PATH" ]; then
      export CUDA_HOME="$(dirname $(dirname $PYTORCH_CUDA_PATH))"
      echo "Setting CUDA_HOME based on PyTorch's CUDA libraries: $CUDA_HOME"
    else
      echo "Warning: Could not determine CUDA_HOME."
      # Last resort - try to use one of common WSL paths
      for path in "/usr/lib/cuda" "/usr/lib/wsl" "/mnt/wslg/usr/lib/wsl"; do
        if [ -d "$path" ]; then
          export CUDA_HOME="$path"
          echo "Trying alternative CUDA_HOME: $CUDA_HOME"
          break
        fi
      done
    fi
  fi
else
  echo "CUDA is not available to PyTorch. This will be very slow without GPU acceleration."
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

# Get the benchmark results
if [ -f "benchmark_results/best_model_config.json" ]; then
  MODEL_NAME=$(python -c "import json; print(json.load(open('benchmark_results/best_model_config.json'))['model_name'])")
  MAX_CONTEXT=$(python -c "import json; print(json.load(open('benchmark_results/best_model_config.json')).get('max_context_length', 2048))")
  SANITIZED_MODEL_NAME=${MODEL_NAME//\//_}
else
  echo "No benchmark results found. Using default values."
  MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
  MAX_CONTEXT=2048
  SANITIZED_MODEL_NAME="Qwen_Qwen2.5-Coder-7B-Instruct"
fi

# Ensure directories exist
mkdir -p logs
mkdir -p trained_models

echo "Starting training for $MODEL_NAME with context length $MAX_CONTEXT..."
echo "CUDA_HOME is set to: $CUDA_HOME"

# Run the training with DeepSpeed disabled or properly configured
python optimized_training.py \
  --model_name "$MODEL_NAME" \
  --context_length $MAX_CONTEXT \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --zero_stage 3 \
  --fp16 \
  --offload_optimizer \
  --offload_param \
  --learning_rate 1e-5 \
  --warmup_steps 100 \
  --epochs 3 \
  --output_dir "./trained_models/${SANITIZED_MODEL_NAME}" \
  2>&1 | tee "logs/optimal_training_log_${SANITIZED_MODEL_NAME}.txt"

echo "Training complete! Model saved to ./trained_models/${SANITIZED_MODEL_NAME}" 