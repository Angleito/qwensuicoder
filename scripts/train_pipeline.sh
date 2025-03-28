#!/bin/bash

# QwenSuiCoder Training Pipeline
# This script runs the entire training pipeline from data preparation to model evaluation

set -e  # Exit on error

# Define directories
ROOT_DIR=$(pwd)
DATA_DIR="$ROOT_DIR/data"
EXAMPLES_DIR="$ROOT_DIR/examples"
CONFIG_DIR="$ROOT_DIR/config"
OUTPUT_DIR="$ROOT_DIR/trained_models"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "QwenSuiCoder Training Pipeline"
echo "========================================================"

# Step 1: Check if Ollama and Qwen are available
echo "Step 1: Checking Ollama and Qwen availability..."
python scripts/check_ollama.py

# Step 2: Process data
echo "Step 2: Processing training data..."
python src/utils/data_processor.py \
  --move-dir "$EXAMPLES_DIR/move" \
  --typescript-dir "$EXAMPLES_DIR/typescript" \
  --python-dir "$EXAMPLES_DIR/python" \
  --raw-data-dir "$DATA_DIR/raw" \
  --processed-data-dir "$DATA_DIR/processed" \
  --training-data-dir "$DATA_DIR/training" \
  --validation-data-dir "$DATA_DIR/validation" \
  --validation-split 0.1

# Step 3: Train the model
echo "Step 3: Training model..."
python src/training/train.py "$CONFIG_DIR/training_config.yaml"

# Step 4: Evaluate the model
echo "Step 4: Evaluating model..."
python src/evaluation/evaluate.py \
  --model-path "$OUTPUT_DIR" \
  --test-file "$DATA_DIR/validation/combined_val.jsonl" \
  --output-file "$OUTPUT_DIR/evaluation_results.json"

echo "========================================================"
echo "Training pipeline completed!"
echo "Model and evaluation results are in $OUTPUT_DIR"
echo "========================================================" 