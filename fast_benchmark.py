#!/usr/bin/env python3
"""
Fast benchmark script targeted only for Qwen2.5-Coder-7B-Instruct
Tests only the target model with 4-bit quantization for speed
"""

import torch
import argparse
import time
import os
import logging
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Only test the target model
TARGET_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

def get_gpu_memory_usage():
    """Get current GPU VRAM usage if CUDA is available."""
    if not torch.cuda.is_available():
        return {"gpu_allocated_gb": 0, "gpu_total_gb": 0}

    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return {"gpu_allocated_gb": allocated, "gpu_total_gb": total}

def clear_memory():
    """Aggressively clear CPU and GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_context_length(model, tokenizer, context_length, device="cuda"):
    """Test if a model can handle a specific context length."""
    try:
        # Create a dummy input with the specified context length
        dummy_input = "a " * (context_length - 10)  # Leave room for special tokens
        inputs = tokenizer(dummy_input, return_tensors="pt").to(device)
        
        # Verify input length
        if inputs.input_ids.shape[1] < context_length * 0.9:
            logger.warning(f"Input too short: {inputs.input_ids.shape[1]} tokens. Tokenization may be efficient.")
        
        # Just forward a few tokens to test memory
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                max_new_tokens=5,
                do_sample=False
            )
        
        return True, None
    except Exception as e:
        return False, str(e)

def fast_benchmark():
    """Run a focused benchmark only on the target model with 4-bit quantization."""
    model_name = TARGET_MODEL
    os.makedirs("benchmark_results", exist_ok=True)
    results = {"model_name": model_name, "context_length": []}
    
    # Check for CUDA
    if not torch.cuda.is_available():
        logger.error("No CUDA GPU detected. This benchmark requires a GPU.")
        return results

    # Log total VRAM
    total_vram = get_gpu_memory_usage()["gpu_total_gb"]
    logger.info(f"Detected CUDA GPU with {total_vram:.2f} GB VRAM")
    
    # Clear memory before starting
    clear_memory()
    logger.info(f"Testing {model_name} with 4-bit quantization...")
    
    # Test model loading with 4-bit quantization
    try:
        # Measure load time and memory usage
        mem_before = get_gpu_memory_usage()["gpu_allocated_gb"]
        start_time = time.time()
        
        # Use 4-bit quantization (NF4 format)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",  # Use auto for better device management
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Measure load time and memory usage
        load_time = time.time() - start_time
        mem_after = get_gpu_memory_usage()["gpu_allocated_gb"]
        vram_usage_gb = mem_after - mem_before
        
        logger.info(f"Model loaded successfully in {load_time:.2f}s. VRAM used: ~{vram_usage_gb:.2f} GB")
        
        # Test various context lengths to find the maximum supported
        context_lengths = [2048, 4096, 8192, 16384, 32768]
        max_supported = None
        
        for ctx_len in context_lengths:
            logger.info(f"Testing context length: {ctx_len}")
            success, error = test_context_length(model, tokenizer, ctx_len)
            
            result = {
                "context_length": ctx_len,
                "success": success,
                "error": error
            }
            results["context_length"].append(result)
            
            if success:
                max_supported = ctx_len
                logger.info(f"Context length {ctx_len} supported ✓")
            else:
                logger.warning(f"Context length {ctx_len} failed: {error}")
                break
        
        # Save best config directly
        config = {
            "model_name": model_name,
            "quantization": "4bit",
            "max_context_length": max_supported or 2048  # Default to 2048 if all tests fail
        }
        
        with open("benchmark_results/best_model_config.json", 'w') as f:
            json.dump(config, f, indent=4)
            
        logger.info(f"Best configuration saved to benchmark_results/best_model_config.json")
        logger.info(f"Model: {model_name}")
        logger.info(f"Quantization: 4bit")
        logger.info(f"Max Context Length: {max_supported or 2048}")
        
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        
        # Save minimal config if test fails
        config = {
            "model_name": model_name,
            "quantization": "4bit",
            "max_context_length": 2048
        }
        
        with open("benchmark_results/best_model_config.json", 'w') as f:
            json.dump(config, f, indent=4)
            
        logger.warning("Saved fallback configuration due to benchmark failure")
    
    # Save detailed results
    with open("benchmark_results/fast_benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info("Fast benchmark complete. Results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast benchmark for Qwen2.5-Coder-7B-Instruct model")
    args = parser.parse_args()
    
    # Run the fast benchmark
    fast_benchmark() 