#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for Qwen 2.5 on Sui Move code and SDK examples.
"""

import os
import yaml
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import datasets
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    base_model: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    train_file: str = field(
        metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."}
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # Parse CLI arguments or load from config file
    if len(os.sys.argv) == 2 and os.sys.argv[1].endswith(".yaml"):
        with open(os.sys.argv[1], "r") as f:
            config = yaml.safe_load(f)
        
        model_args = ModelArguments(
            base_model=config["model"]["base_model"],
            use_lora=config["model"]["use_lora"],
        )
        
        data_args = DataArguments(
            train_file=config["data"]["train_file"],
            validation_file=config["data"]["validation_file"],
            max_seq_length=config["data"]["max_seq_length"],
        )
        
        # Map training config to HF TrainingArguments
        train_args_dict = {
            "output_dir": config["model"]["output_dir"],
            "per_device_train_batch_size": config["training"]["train_batch_size"],
            "per_device_eval_batch_size": config["training"]["eval_batch_size"],
            "num_train_epochs": config["training"]["num_train_epochs"],
            "learning_rate": config["training"]["learning_rate"],
            "lr_scheduler_type": config["training"]["lr_scheduler_type"],
            "warmup_ratio": config["training"]["warmup_ratio"],
            "weight_decay": config["training"]["weight_decay"],
            "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
            "gradient_checkpointing": config["training"]["gradient_checkpointing"],
            "fp16": config["training"]["fp16"],
            "bf16": config["training"]["bf16"],
            "max_grad_norm": config["training"]["max_grad_norm"],
            "logging_steps": config["training"]["logging_steps"],
            "evaluation_strategy": "steps",
            "eval_steps": config["training"]["eval_steps"],
            "save_strategy": "steps",
            "save_steps": config["training"]["save_steps"],
            "save_total_limit": config["training"]["save_total_limit"],
            "report_to": "tensorboard",
        }
        training_args = TrainingArguments(**train_args_dict)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Training arguments: {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load datasets
    logger.info(f"Loading datasets from {data_args.train_file} and {data_args.validation_file}")
    train_dataset = datasets.load_dataset(
        "json", 
        data_files=data_args.train_file, 
        split="train"
    )
    
    if data_args.validation_file:
        eval_dataset = datasets.load_dataset(
            "json", 
            data_files=data_args.validation_file, 
            split="train"
        )
    else:
        eval_dataset = None

    # Load model and tokenizer
    logger.info(f"Loading model {model_args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model, trust_remote_code=True)
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Configure LoRA if enabled
    if model_args.use_lora:
        logger.info("Configuring model for LoRA training")
        
        # Load LoRA configuration from config
        with open(os.sys.argv[1], "r") as f:
            config = yaml.safe_load(f)
        
        lora_config = LoraConfig(
            r=config["model"]["lora_config"]["r"],
            lora_alpha=config["model"]["lora_config"]["lora_alpha"],
            lora_dropout=config["model"]["lora_config"]["lora_dropout"],
            bias=config["model"]["lora_config"]["bias"],
            task_type=config["model"]["lora_config"]["task_type"],
            target_modules=config["model"]["lora_config"]["target_modules"],
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Initialize SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=data_args.max_seq_length,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Output training metrics
    training_summary = {"train_runtime": trainer.state.total_flos}
    logger.info(f"Training complete! Summary: {training_summary}")

if __name__ == "__main__":
    main() 