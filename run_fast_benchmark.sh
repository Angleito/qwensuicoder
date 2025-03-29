#!/bin/bash

# Fast benchmark script focused only on the Qwen2.5-Coder-7B-Instruct model
echo "Running fast benchmark for Qwen/Qwen2.5-Coder-7B-Instruct..."

# Ensure directories exist
mkdir -p benchmark_results
mkdir -p logs

# Kill existing benchmark process if running
existing_pid=$(ps aux | grep benchmark_models.py | grep -v grep | awk '{print $2}')
if [ ! -z "$existing_pid" ]; then
    echo "Stopping existing benchmark process (PID: $existing_pid)..."
    kill $existing_pid
    sleep 3  # Wait for process to terminate
fi

# Clear GPU memory before starting
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

# Run the fast benchmark
python fast_benchmark.py 2>&1 | tee logs/fast_benchmark.log

# Check if benchmark was successful
if [ $? -ne 0 ]; then
    echo "Error: Fast benchmark failed. Check logs/fast_benchmark.log"
    exit 1
fi

# Generate the optimal training script based on benchmark results
echo "Creating optimized training script..."
cat > run_optimal_training.sh << EOL
#!/bin/bash

# Generated by fast benchmark
# Optimized for Qwen/Qwen2.5-Coder-7B-Instruct with 4-bit quantization

# Ensure directories exist
mkdir -p logs
mkdir -p trained_models

# Get configuration from benchmark
MODEL_NAME=\$(python -c "import json; print(json.load(open('benchmark_results/best_model_config.json'))['model_name'])")
MAX_CONTEXT=\$(python -c "import json; print(json.load(open('benchmark_results/best_model_config.json')).get('max_context_length', 2048))")
SANITIZED_MODEL_NAME=\${MODEL_NAME//\\//_}

echo "Starting training for \$MODEL_NAME with context length \$MAX_CONTEXT..."

# Special handling for WSL environment where CUDA may be available to PyTorch
# but NVCC compiler might be missing
echo "Checking CUDA environment..."
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
  echo "CUDA is available to PyTorch"
  
  # Skip CUDA extension builds for DeepSpeed by setting these env vars
  export DS_BUILD_CPU_ADAM=1
  export DS_BUILD_FUSED_ADAM=0
  export DS_BUILD_UTILS=0
  export DS_BUILD_TRANSFORMER=0
  export DS_BUILD_TRANSFORMER_INFERENCE=0
  export DS_BUILD_STOCHASTIC_TRANSFORMER=0
  export DS_BUILD_SPARSE_ATTN=0
  export TORCH_CUDA_ARCH_LIST="8.6"
  
  # Set CUDA_HOME if we can find it
  if [ -d "/usr/lib/wsl/lib" ]; then
    echo "Using WSL-specific CUDA paths"
    export CUDA_HOME="/usr/lib/wsl/lib"
  else
    # Try to find where PyTorch's CUDA libraries are
    PYTORCH_CUDA_PATH=\$(python -c "import torch; print(torch._C._cuda_getDirFor('nvrtc'))" 2>/dev/null || echo "")
    if [ ! -z "\$PYTORCH_CUDA_PATH" ]; then
      export CUDA_HOME="\$(dirname \$(dirname \$PYTORCH_CUDA_PATH))"
      echo "Setting CUDA_HOME based on PyTorch's CUDA libraries: \$CUDA_HOME"
    else
      echo "Warning: Could not determine CUDA_HOME. Using CPU-only fallback."
    fi
  fi
else
  echo "CUDA is not available to PyTorch. Using CPU only."
fi

# Run optimized training with the benchmarked parameters
python optimized_training.py \\
  --model_name "\$MODEL_NAME" \\
  --context_length \$MAX_CONTEXT \\
  --batch_size 1 \\
  --gradient_accumulation_steps 16 \\
  --zero_stage 3 \\
  --fp16 \\
  --offload_optimizer \\
  --offload_param \\
  --learning_rate 1e-5 \\
  --warmup_steps 100 \\
  --epochs 3 \\
  --output_dir "./trained_models/\${SANITIZED_MODEL_NAME}" \\
  2>&1 | tee "logs/optimal_training_log_\${SANITIZED_MODEL_NAME}.txt"

echo "Training complete! Model saved to ./trained_models/\${SANITIZED_MODEL_NAME}"
EOL

# Make the script executable
chmod +x run_optimal_training.sh

echo "Fast benchmark complete!"
echo "Run './run_optimal_training.sh' to start training with the optimal configuration" 