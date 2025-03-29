#!/usr/bin/env python3
"""
Simple demonstration script to load a small LLM and generate text
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import argparse
import time

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple LLM demo")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model to load for demonstration")
    parser.add_argument("--prompt", type=str, 
                        default="Write a function to check if a string is a palindrome.",
                        help="Prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=500,
                       help="Maximum number of tokens to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info(f"Loading model: {args.model_name}")
    start_time = time.time()
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        
        # For a smaller model that can run on CPU
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Format prompt properly
        if "TinyLlama" in args.model_name:
            formatted_prompt = f"<|system|>\nYou are a helpful coding assistant.\n<|user|>\n{args.prompt}\n<|assistant|>"
        elif "Qwen" in args.model_name:
            formatted_prompt = f"<|im_start|>system\nYou are a helpful coding assistant.\n<|im_end|>\n<|im_start|>user\n{args.prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"Write a {args.prompt}"
            
        logger.info(f"Generating response for prompt: {args.prompt}")
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate
        generate_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        generate_time = time.time() - generate_start
        
        # Decode output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generation completed in {generate_time:.2f} seconds")
        logger.info("\n" + "="*50 + "\nGENERATED RESPONSE:\n" + "="*50)
        print(response)
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full error details:")

if __name__ == "__main__":
    main()
