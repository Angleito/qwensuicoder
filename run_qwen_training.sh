#!/bin/bash

# Qwen Coder 14B Benchmark and Training Script
# This script runs benchmarks to find optimal settings, then trains the model

set -e  # Exit on error

# Colors for nice output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found"
    exit 1
fi

# Check for required packages
python3 -c "import torch, transformers, datasets, deepspeed, peft" || {
    echo "Installing required packages..."
    pip install -q torch transformers datasets accelerate deepspeed peft matplotlib
}

# Check for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
    if [ -f ~/.huggingface/token ]; then
        export HF_TOKEN=$(cat ~/.huggingface/token)
        echo -e "${GREEN}Using Hugging Face token from ~/.huggingface/token${NC}"
    else
        echo -e "${YELLOW}WARNING: HF_TOKEN environment variable not set.${NC}"
        echo "You might need to set it for accessing gated models:"
        echo "export HF_TOKEN=your_token_here"
    fi
fi

# Create output directories
mkdir -p benchmark_results
mkdir -p trained_models/qwen_coder_14b

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}=== STEP 1: Running Qwen 2.5 Benchmarks ===${NC}"
echo -e "${GREEN}============================================${NC}"

# Run benchmarks
python3 benchmark_qwen_coder.py \
    --output_dir benchmark_results \
    --device auto

# Check if benchmark was successful
if [ -f "benchmark_results/optimal_settings.json" ]; then
    echo -e "${GREEN}Benchmark completed successfully!${NC}"
else
    echo -e "${YELLOW}WARNING: Benchmark did not generate optimal settings.${NC}"
    echo "Will use default training settings."
fi

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}=== STEP 2: Training Qwen 2.5 Coder 14B ===${NC}"
echo -e "${GREEN}============================================${NC}"

# Run training with benchmark results
python3 train_qwen_coder.py \
    --model_name "Qwen/Qwen2.5-14B-Coder" \
    --train_file "data/training/combined_train.jsonl" \
    --output_dir "trained_models/qwen_coder_14b" \
    --use_lora \
    --auto_optimize \
    --settings_file "benchmark_results/optimal_settings.json" \
    --deepspeed \
    --fp16

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}=== Training completed! ===${NC}"
echo -e "${GREEN}============================================${NC}"
echo "Trained model saved to: trained_models/qwen_coder_14b"
echo "To use the model:"
echo "from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "tokenizer = AutoTokenizer.from_pretrained('trained_models/qwen_coder_14b')"
echo "model = AutoModelForCausalLM.from_pretrained('trained_models/qwen_coder_14b')" 