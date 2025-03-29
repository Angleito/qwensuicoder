"""
Optimized training script for Qwen 2.5 Coder 14B
Uses DeepSpeed for memory optimization with system resource maximization
"""

import os
import torch
import argparse
import json
import logging
import psutil
import deepspeed
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_system_resources():
    """Get available system resources for optimization"""
    resources = {}
    
    # CPU
    resources["cpu_count"] = psutil.cpu_count(logical=True)
    
    # RAM
    ram = psutil.virtual_memory()
    resources["ram_total_gb"] = ram.total / (1024**3)
    resources["ram_available_gb"] = ram.available / (1024**3)
    
    # GPU
    if torch.cuda.is_available():
        resources["gpu_available"] = True
        resources["gpu_count"] = torch.cuda.device_count()
        resources["gpu_name"] = torch.cuda.get_device_name(0)
        resources["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        resources["gpu_available"] = False
    
    return resources

def create_deepspeed_config(args, resources):
    """Create an optimized DeepSpeed configuration based on system resources"""
    
    config = {
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": args.fp16,
        },
        "zero_optimization": {
            "stage": args.zero_stage,
            "offload_optimizer": {
                "device": "cpu" if args.offload_optimizer else "none"
            },
            "offload_param": {
                "device": "cpu" if args.offload_param else "none"
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps
            }
        }
    }
    
    # Optimize for hardware
    if resources["gpu_available"]:
        # For larger VRAM GPUs
        if resources.get("gpu_memory_gb", 0) >= 24:
            # We can use a higher batch size and less aggressive optimization
            if args.auto_optimize:
                config["zero_optimization"]["stage"] = 2
                config["zero_optimization"]["offload_optimizer"]["device"] = "none"
                config["zero_optimization"]["offload_param"]["device"] = "none"
        # For mid-range GPUs (8-16GB)
        elif resources.get("gpu_memory_gb", 0) >= 8:
            # ZeRO-3 with optimizer offloading
            if args.auto_optimize:
                config["zero_optimization"]["stage"] = 3
                config["zero_optimization"]["offload_optimizer"]["device"] = "cpu"
                config["zero_optimization"]["offload_param"]["device"] = "none"
        # For smaller GPUs
        else:
            # Most aggressive optimization
            if args.auto_optimize:
                config["zero_optimization"]["stage"] = 3
                config["zero_optimization"]["offload_optimizer"]["device"] = "cpu"
                config["zero_optimization"]["offload_param"]["device"] = "cpu"
    else:
        # CPU-only mode
        if args.auto_optimize:
            config["zero_optimization"]["stage"] = 3
            
    return config

def load_datasets(args):
    """Load and prepare datasets for training"""
    # Load datasets
    if os.path.exists(args.train_file):
        logger.info(f"Loading dataset from {args.train_file}")
        dataset = load_dataset('json', data_files={'train': args.train_file})
    else:
        # Fallback to a demonstration dataset if file doesn't exist
        logger.warning(f"Training file {args.train_file} not found. Using a small demo dataset.")
        
        # Create a small demo dataset
        demo_data = []
        for i in range(10):
            demo_data.append({
                "instruction": "Implement a simple counter module in Sui Move",
                "input": "",
                "output": "module counter {\n    use sui::object::{Self, UID};\n    use sui::tx_context::{Self, TxContext};\n    use sui::transfer;\n\n    struct Counter has key {\n        id: UID,\n        value: u64,\n    }\n\n    public fun create(ctx: &mut TxContext) {\n        let counter = Counter {\n            id: object::new(ctx),\n            value: 0,\n        };\n        transfer::share_object(counter);\n    }\n\n    public fun increment(counter: &mut Counter) {\n        counter.value = counter.value + 1;\n    }\n\n    public fun value(counter: &Counter): u64 {\n        counter.value\n    }\n}"
            })
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            for item in demo_data:
                f.write(json.dumps(item) + '\n')
            demo_file = f.name
        
        dataset = load_dataset('json', data_files={'train': demo_file})
    
    # Process datasets
    def process_dataset(examples):
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"], 
            examples.get("input", [""] * len(examples["instruction"])), 
            examples["output"]
        ):
            if input_text:
                text = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            else:
                text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            texts.append(text)
        
        return {"text": texts}
    
    # Check if the dataset has the expected format
    required_columns = ["instruction", "output"]
    has_required_columns = all(col in dataset["train"].column_names for col in required_columns)
    
    if has_required_columns:
        processed_dataset = dataset.map(
            process_dataset, 
            batched=True, 
            remove_columns=dataset["train"].column_names
        )
    else:
        # Fallback for datasets with just text
        logger.warning("Dataset doesn't have instruction/output format. Using as-is.")
        if "text" not in dataset["train"].column_names:
            raise ValueError("Dataset must have either 'instruction'/'output' columns or a 'text' column.")
        processed_dataset = dataset
    
    return processed_dataset

def train():
    parser = argparse.ArgumentParser(description="Train Qwen 2.5 Coder 14B with optimized settings")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Coder",
                       help="Qwen model to fine-tune")
    parser.add_argument("--train_file", type=str, default="data/training/combined_train.jsonl",
                       help="Path to training data file")
    parser.add_argument("--output_dir", type=str, default="./trained_models/qwen_coder_14b",
                       help="Directory to save the trained model")
    parser.add_argument("--context_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout probability")
    
    # Quantization arguments
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit precision")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Number of steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Learning rate warmup steps")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging interval in steps")
    parser.add_argument("--save_steps", type=int, default=200,
                       help="Checkpoint saving interval")
    
    # DeepSpeed arguments
    parser.add_argument("--deepspeed", action="store_true",
                       help="Use DeepSpeed for training")
    parser.add_argument("--zero_stage", type=int, default=3,
                       help="ZeRO optimization stage (0, 1, 2, or 3)")
    parser.add_argument("--offload_optimizer", action="store_true",
                       help="Offload optimizer states to CPU")
    parser.add_argument("--offload_param", action="store_true",
                       help="Offload parameters to CPU")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    
    # Auto optimization
    parser.add_argument("--auto_optimize", action="store_true",
                       help="Automatically optimize settings based on hardware")
    parser.add_argument("--settings_file", type=str, default="",
                       help="Path to benchmark settings JSON file")
    
    args = parser.parse_args()
    
    # Get system resources
    resources = get_system_resources()
    logger.info("=== System Resources ===")
    logger.info(f"CPU: {resources['cpu_count']} cores")
    logger.info(f"RAM: {resources['ram_total_gb']:.2f} GB")
    if resources["gpu_available"]:
        logger.info(f"GPU: {resources['gpu_name']}")
        logger.info(f"GPU Memory: {resources['gpu_memory_gb']:.2f} GB")
    else:
        logger.info("No GPU detected. Training will be slow.")
    
    # Load optimal settings from benchmark if available
    if args.settings_file and os.path.exists(args.settings_file):
        logger.info(f"Loading optimal settings from {args.settings_file}")
        with open(args.settings_file, 'r') as f:
            settings = json.load(f)
            
        if args.auto_optimize:
            if "quantization" in settings:
                if settings["quantization"] == "4bit":
                    args.load_in_4bit = True
                    args.load_in_8bit = False
                elif settings["quantization"] == "8bit":
                    args.load_in_4bit = False
                    args.load_in_8bit = True
                else:  # 16bit
                    args.load_in_4bit = False
                    args.load_in_8bit = False
                    
            if "context_length" in settings:
                args.context_length = settings["context_length"]
                
            if "batch_size" in settings:
                args.batch_size = settings["batch_size"]
                
            if "gradient_accumulation_steps" in settings:
                args.gradient_accumulation_steps = settings["gradient_accumulation_steps"]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create DeepSpeed config
    if args.deepspeed:
        ds_config = create_deepspeed_config(args, resources)
        with open(os.path.join(args.output_dir, 'ds_config.json'), 'w') as f:
            json.dump(ds_config, f, indent=2)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Handle missing chat template if needed
    if not tokenizer.chat_template:
        default_chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}\n<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'system' %}\n<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% endif %}\n{% endfor %}"
        tokenizer.chat_template = default_chat_template
    
    # Ensure padding token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimized settings
    logger.info(f"Loading model: {args.model_name}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" 
    }
    
    if args.load_in_4bit:
        logger.info("Loading model in 4-bit precision")
        model_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4"
        })
    elif args.load_in_8bit:
        logger.info("Loading model in 8-bit precision")
        model_kwargs.update({
            "load_in_8bit": True
        })
    else:
        logger.info("Loading model in 16-bit precision")
        model_kwargs.update({
            "torch_dtype": torch.float16
        })
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    
    # Apply LoRA if enabled
    if args.use_lora:
        logger.info("Setting up LoRA fine-tuning")
        
        # Prepare model for k-bit training if using quantization
        if args.load_in_4bit or args.load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", 
                "k_proj", 
                "v_proj", 
                "o_proj"
            ]
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load datasets
    train_dataset = load_datasets(args)["train"]
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=args.fp16,
        push_to_hub=False,
        disable_tqdm=False,
        dataloader_num_workers=min(8, resources["cpu_count"] // 2),
        gradient_checkpointing=True,
        deepspeed=os.path.join(args.output_dir, 'ds_config.json') if args.deepspeed else None
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Configure Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train model
    logger.info("Starting training")
    train_result = trainer.train()
    
    # Log training results
    logger.info(f"Training completed with {train_result.metrics}")
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    train() 