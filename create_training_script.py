#!/usr/bin/env python3
"""
Script to create training script for Qwen model with TypeScript libraries
"""

import os

# Create the training script
script_content = """#!/bin/bash

# No-CUDA training script that completely bypasses DeepSpeed's CUDA detection
# while still allowing PyTorch to use CUDA for model acceleration

echo "Setting up training without DeepSpeed CUDA extensions..."

# Force DeepSpeed to skip all CUDA extension building
export DS_BUILD_OPS=0
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_UTILS=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_TRANSFORMER_INFERENCE=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0
export DS_BUILD_SPARSE_ATTN=0
export TORCH_CUDA_ARCH_LIST="8.6"
# This is critical - disable ops completely
export DISABLE_CONFLUENCE=1
export DISABLE_AMOS=1

# Check if CUDA is available through PyTorch
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
  echo "✓ CUDA is available to PyTorch (will be used for tensor operations)"
  echo "Deliberately not setting CUDA_HOME to avoid DeepSpeed CUDA extension builds"
else
  echo "⚠ CUDA is not available to PyTorch. This will be very slow without GPU acceleration."
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
  SANITIZED_MODEL_NAME=${MODEL_NAME//\\//_}
else
  echo "No benchmark results found. Using default values."
  MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
  MAX_CONTEXT=2048
  SANITIZED_MODEL_NAME="Qwen_Qwen2.5-Coder-7B-Instruct"
fi

# Ensure directories exist
mkdir -p logs
mkdir -p trained_models
mkdir -p training_data

# Add Bluefin and Cetus TypeScript libraries to training data
echo "Adding TypeScript libraries to training data..."

# Create a file with Bluefin TS SDK code
echo "Adding Bluefin TS SDK to training data..."
find bluefin-v2-client-ts -name "*.ts" -o -name "*.tsx" 2>/dev/null | xargs cat > training_data/bluefin_ts_sdk.txt 2>/dev/null || echo "Warning: Could not find Bluefin TS SDK files"

# Create a file with Cetus Protocol code
echo "Adding Cetus Protocol to training data..."
find node_modules/@cetusprotocol -name "*.ts" -o -name "*.js" 2>/dev/null | xargs cat > training_data/cetus_protocol.txt 2>/dev/null || echo "Warning: Could not find Cetus Protocol files"

# Count lines added to training data
BLUEFIN_LINES=$(wc -l < training_data/bluefin_ts_sdk.txt 2>/dev/null || echo "0")
CETUS_LINES=$(wc -l < training_data/cetus_protocol.txt 2>/dev/null || echo "0")
echo "Added $BLUEFIN_LINES lines of Bluefin code and $CETUS_LINES lines of Cetus code to training data"

# Run optimized training with custom training data
echo "Starting training for $MODEL_NAME with context length $MAX_CONTEXT..."
python optimized_training.py \\
  --model_name "$MODEL_NAME" \\
  --context_length $MAX_CONTEXT \\
  --batch_size 1 \\
  --gradient_accumulation_steps 16 \\
  --zero_stage 3 \\
  --fp16 \\
  --offload_optimizer \\
  --offload_param \\
  --learning_rate 1e-5 \\
  --warmup_steps 100 \\
  --epochs 3 \\
  --output_dir "./trained_models/${SANITIZED_MODEL_NAME}" \\
  --training_data_dir "./training_data" \\
  2>&1 | tee "logs/training_log_with_ts_sdks.txt"

echo "Training complete! Model saved to ./trained_models/${SANITIZED_MODEL_NAME}"
echo "Log file saved to logs/training_log_with_ts_sdks.txt"
"""

# Write the script to a file
with open('run_training_no_cuda.sh', 'w') as f:
    f.write(script_content)

# Make the script executable
os.chmod('run_training_no_cuda.sh', 0o755)

print("Created run_training_no_cuda.sh - run it to train with TypeScript libraries") 