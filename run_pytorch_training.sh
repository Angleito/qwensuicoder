#!/bin/bash

# PyTorch-only training script that completely avoids DeepSpeed
# This is a simplified training approach for WSL environments with CUDA integration

echo "Setting up PyTorch-only training (no DeepSpeed)..."

# Ensure directories exist
mkdir -p logs
mkdir -p trained_models
mkdir -p training_data

# Prepare training data directory with Bluefin and Cetus libraries
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

# Create a simple PyTorch training script
cat > pytorch_training.py << EOF
#!/usr/bin/env python3
"""
Simple PyTorch-only training script for fine-tuning Qwen2.5-Coder-7B-Instruct
without using DeepSpeed (to avoid CUDA dependency issues in WSL).
"""

import os
import argparse
import logging
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class TypeScriptDataset(Dataset):
    """Dataset for TypeScript libraries."""
    
    def __init__(self, tokenizer, data_dir: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load TypeScript examples from data directory
        if os.path.isdir(data_dir):
            txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
            for txt_file in txt_files:
                try:
                    logger.info(f"Loading {os.path.basename(txt_file)}")
                    with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    
                    # Split content into manageable chunks
                    chunk_size = 5000
                    content_chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                    
                    # Create examples
                    for chunk in content_chunks:
                        # Instruction format
                        example = {
                            "text": f"<|im_start|>user\\nExplain this TypeScript code:\\n\\n\`\`\`typescript\\n{chunk}\\n\`\`\`<|im_end|>\\n<|im_start|>assistant\\nLet me explain this TypeScript code step by step:\\n\\nThis code is part of a library. The main functionality includes:\\n\\n1. Types and interfaces for TypeScript development\\n2. Functions for handling data structures and operations\\n3. Implementation of key utility methods\\n\\nThe code follows TypeScript best practices with proper typing and modular structure.\\n<|im_end|>"
                        }
                        self.examples.append(example)
                    
                    logger.info(f"Added {len(content_chunks)} examples from {os.path.basename(txt_file)}")
                except Exception as e:
                    logger.warning(f"Error processing {txt_file}: {e}")
        
        # Add some generic coding examples too
        generic_examples = [
            {"text": "<|im_start|>user\\nWrite a function to check if a string is a palindrome<|im_end|>\\n<|im_start|>assistant\\ndef is_palindrome(s):\\n    s = s.lower().replace(' ', '')\\n    return s == s[::-1]\\n<|im_end|>"},
            {"text": "<|im_start|>user\\nImplement a binary search function<|im_end|>\\n<|im_start|>assistant\\ndef binary_search(arr, target):\\n    left, right = 0, len(arr) - 1\\n    while left <= right:\\n        mid = (left + right) // 2\\n        if arr[mid] == target:\\n            return mid\\n        elif arr[mid] < target:\\n            left = mid + 1\\n        else:\\n            right = mid - 1\\n    return -1\\n<|im_end|>"}
        ]
        
        self.examples.extend(generic_examples * 10)  # Add 10 copies of each generic example
        logger.info(f"Total examples: {len(self.examples)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]["text"]
        
        tokenized = self.tokenizer(
            example,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        
        # Create labels (mask user prompt with -100)
        labels = input_ids.clone()
        prompt_end = example.find("<|im_end|>\\n<|im_start|>assistant")
        if prompt_end != -1:
            tokenized_prompt = self.tokenizer(
                example[:prompt_end], 
                return_tensors="pt"
            )
            prompt_len = len(tokenized_prompt.input_ids[0])
            labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Simple PyTorch training for Qwen2.5")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Model name")
    parser.add_argument("--context_length", type=int, default=2048, help="Maximum context length")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--training_data_dir", type=str, default="./training_data", help="Directory with training data")
    parser.add_argument("--output_dir", type=str, default="./trained_models", help="Output directory for model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        trust_remote_code=True,
        device_map="auto"  # Let PyTorch decide optimal device mapping
    )
    
    # Load dataset
    logger.info(f"Loading TypeScript dataset from {args.training_data_dir}")
    train_dataset = TypeScriptDataset(
        tokenizer=tokenizer,
        data_dir=args.training_data_dir,
        max_length=args.context_length
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        fp16=args.fp16,
        report_to="none",  # Disable wandb/tensorboard
        disable_tqdm=False,
        save_total_limit=1,  # Keep only the last checkpoint
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x pytorch_training.py

# Set up model name
MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR="./trained_models/$(echo $MODEL_NAME | tr '/' '_')"

# Run the PyTorch training script
echo "Starting PyTorch training for $MODEL_NAME..."
python pytorch_training.py \
  --model_name "$MODEL_NAME" \
  --context_length 2048 \
  --batch_size 1 \
  --learning_rate 2e-5 \
  --num_epochs 1 \
  --gradient_accumulation_steps 16 \
  --fp16 \
  --training_data_dir "./training_data" \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "logs/pytorch_training_log.txt"

echo "Training complete! Model saved to $OUTPUT_DIR"
echo "Log file saved to logs/pytorch_training_log.txt" 