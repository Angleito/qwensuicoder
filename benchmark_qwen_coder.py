"""
Benchmark script for Qwen 2.5 Coder 14B
Tests hardware compatibility and finds optimal training settings
"""

import torch
import argparse
import time
import os
import logging
import psutil
import json
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gc
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Qwen model to test
QWEN_MODEL = "Qwen/Qwen2.5-14B-Coder"

# Context lengths to test (in tokens)
CONTEXT_LENGTHS = [1024, 2048, 4096, 8192]

def get_memory_usage():
    """Get memory usage information"""
    mem_info = {}
    
    # RAM usage
    ram = psutil.virtual_memory()
    mem_info["ram_used_gb"] = ram.used / (1024**3)
    mem_info["ram_total_gb"] = ram.total / (1024**3)
    mem_info["ram_percent"] = ram.percent
    
    # CPU info
    mem_info["cpu_count"] = psutil.cpu_count(logical=True)
    mem_info["cpu_usage"] = psutil.cpu_percent(interval=1, percpu=False)
    
    # GPU usage if available
    if torch.cuda.is_available():
        mem_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        mem_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        mem_info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        mem_info["gpu_name"] = torch.cuda.get_device_name(0)
    
    return mem_info

def test_model_loading(device="auto", use_4bit=False, use_8bit=False):
    """Test if the Qwen model can be loaded with the given device configuration"""
    logger.info(f"Testing model: {QWEN_MODEL} on device: {device}")
    logger.info(f"Using 4-bit quantization: {use_4bit}, 8-bit quantization: {use_8bit}")
    
    try:
        # Try loading just the config first to get model size
        config = AutoConfig.from_pretrained(QWEN_MODEL)
        
        # Get parameter count
        if hasattr(config, "num_parameters"):
            num_params = config.num_parameters
        else:
            # Rough estimation based on hidden size and layers
            hidden_size = getattr(config, "hidden_size", 0)
            n_layers = getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0)
            vocab_size = getattr(config, "vocab_size", 0)
            
            if hidden_size and n_layers and vocab_size:
                # Very rough estimate: 12 * hidden_size^2 * n_layers + vocab_size * hidden_size
                num_params = 12 * (hidden_size ** 2) * n_layers + vocab_size * hidden_size
            else:
                num_params = 0
        
        logger.info(f"Model has approximately {num_params / 1e9:.2f}B parameters")
        
        # Record memory before loading
        mem_before = get_memory_usage()
        
        # Attempt to load the model
        start_time = time.time()
        
        kwargs = {
            "device_map": device,
            "low_cpu_mem_usage": True,
        }
        
        if use_4bit:
            kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4"
            })
        elif use_8bit:
            kwargs.update({
                "load_in_8bit": True
            })
        else:
            kwargs.update({
                "torch_dtype": torch.float16
            })
        
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            **kwargs
        )
        load_time = time.time() - start_time
        
        # Record memory after loading
        mem_after = get_memory_usage()
        
        # Calculate memory used
        mem_used = {
            "ram_used_gb": mem_after["ram_used_gb"] - mem_before["ram_used_gb"],
            "gpu_allocated_gb": mem_after.get("gpu_allocated_gb", 0) - mem_before.get("gpu_allocated_gb", 0),
        }
        
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Additional RAM used: {mem_used['ram_used_gb']:.2f} GB")
        if torch.cuda.is_available():
            logger.info(f"Additional GPU memory used: {mem_used['gpu_allocated_gb']:.2f} GB")
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "success": True,
            "model_name": QWEN_MODEL,
            "parameters": num_params / 1e9,
            "load_time_seconds": load_time,
            "memory_usage": mem_used,
            "quantization": "4bit" if use_4bit else "8bit" if use_8bit else "16bit"
        }
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {
            "success": False,
            "model_name": QWEN_MODEL,
            "error": str(e),
            "quantization": "4bit" if use_4bit else "8bit" if use_8bit else "16bit"
        }

def test_context_length(context_length, device="auto", use_4bit=False, use_8bit=False):
    """Test if the model can handle the specified context length"""
    logger.info(f"Testing {QWEN_MODEL} with context length {context_length}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        
        # Load model
        kwargs = {
            "device_map": device,
            "low_cpu_mem_usage": True,
        }
        
        if use_4bit:
            kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4"
            })
        elif use_8bit:
            kwargs.update({
                "load_in_8bit": True
            })
        else:
            kwargs.update({
                "torch_dtype": torch.float16
            })
        
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            **kwargs
        )
        
        # Create input of specified context length
        input_ids = torch.ones((1, context_length), dtype=torch.long)
        
        if device == "cuda" and torch.cuda.is_available():
            input_ids = input_ids.cuda()
        elif device != "auto":
            input_ids = input_ids.to(device)
        
        # Record memory before inference
        mem_before = get_memory_usage()
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        inference_time = time.time() - start_time
        
        # Record memory after inference
        mem_after = get_memory_usage()
        
        # Calculate memory used
        mem_used = {
            "ram_used_gb": mem_after["ram_used_gb"] - mem_before["ram_used_gb"],
            "gpu_allocated_gb": mem_after.get("gpu_allocated_gb", 0) - mem_before.get("gpu_allocated_gb", 0),
        }
        
        tokens_per_second = context_length / inference_time if inference_time > 0 else 0
        
        logger.info(f"Context length {context_length} successful")
        logger.info(f"Inference time: {inference_time:.2f} seconds ({tokens_per_second:.2f} tokens/sec)")
        logger.info(f"Additional RAM used: {mem_used['ram_used_gb']:.2f} GB")
        if torch.cuda.is_available():
            logger.info(f"Additional GPU memory used: {mem_used['gpu_allocated_gb']:.2f} GB")
        
        # Clean up
        del model, tokenizer, outputs, input_ids
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "success": True,
            "model_name": QWEN_MODEL,
            "context_length": context_length,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_usage": mem_used,
            "quantization": "4bit" if use_4bit else "8bit" if use_8bit else "16bit"
        }
    
    except Exception as e:
        logger.error(f"Failed with context length {context_length}: {e}")
        return {
            "success": False,
            "model_name": QWEN_MODEL,
            "context_length": context_length,
            "error": str(e),
            "quantization": "4bit" if use_4bit else "8bit" if use_8bit else "16bit"
        }

def visualize_results(results, output_dir):
    """Create visualizations of benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for visualization
    model_loading_results = results["model_loading"]
    context_length_results = results["context_length"]
    
    # Plot loading times for different quantization methods
    quantization_methods = []
    load_times = []
    ram_usages = []
    gpu_usages = []
    
    for result in model_loading_results:
        if result["success"]:
            quantization_methods.append(result["quantization"])
            load_times.append(result["load_time_seconds"])
            ram_usages.append(result["memory_usage"]["ram_used_gb"])
            if "gpu_allocated_gb" in result["memory_usage"]:
                gpu_usages.append(result["memory_usage"]["gpu_allocated_gb"])
            else:
                gpu_usages.append(0)
    
    # Plot model loading times by quantization
    plt.figure(figsize=(10, 6))
    plt.bar(quantization_methods, load_times)
    plt.title("Model Loading Times by Quantization")
    plt.xlabel("Quantization Method")
    plt.ylabel("Loading Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_loading_times.png"))
    
    # Plot memory usage by quantization
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(quantization_methods))
    plt.bar(x - width/2, ram_usages, width, label="RAM Usage (GB)")
    
    if any(gpu_usages):
        plt.bar(x + width/2, gpu_usages, width, label="GPU Usage (GB)")
    
    plt.title("Memory Usage by Quantization")
    plt.xlabel("Quantization Method")
    plt.ylabel("Memory (GB)")
    plt.xticks(x, quantization_methods)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_usage_by_quant.png"))
    
    # Extract context length performance data
    context_lengths = []
    inference_times = []
    tokens_per_second = []
    mem_usages = []
    
    for result in context_length_results:
        if result["success"]:
            context_lengths.append(result["context_length"])
            inference_times.append(result["inference_time_seconds"])
            tokens_per_second.append(result["tokens_per_second"])
            mem_usages.append(result["memory_usage"].get("gpu_allocated_gb", 0) or result["memory_usage"]["ram_used_gb"])
    
    # Plot inference time by context length
    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths, inference_times, 'o-', linewidth=2)
    plt.title("Inference Time by Context Length")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Inference Time (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_time.png"))
    
    # Plot tokens per second by context length
    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths, tokens_per_second, 'o-', linewidth=2)
    plt.title("Throughput by Context Length")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Tokens per Second")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tokens_per_second.png"))
    
    # Plot memory usage by context length
    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths, mem_usages, 'o-', linewidth=2)
    plt.title("Memory Usage by Context Length")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Memory Usage (GB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_by_context.png"))

def get_optimal_training_settings(results):
    """Determine optimal training settings based on benchmark results"""
    
    # Default conservative settings
    optimal_settings = {
        "quantization": "4bit",
        "context_length": 1024,
        "batch_size": 1,
        "gradient_accumulation_steps": 16
    }
    
    # Find the best quantization method that successfully loaded
    successful_loading = [r for r in results["model_loading"] if r["success"]]
    if successful_loading:
        # Prefer higher precision if available (16bit > 8bit > 4bit)
        if any(r["quantization"] == "16bit" for r in successful_loading):
            optimal_settings["quantization"] = "16bit"
        elif any(r["quantization"] == "8bit" for r in successful_loading):
            optimal_settings["quantization"] = "8bit"
    
    # Find the largest context length that worked
    successful_context = [r for r in results["context_length"] if r["success"]]
    if successful_context:
        # Sort by context length (descending)
        successful_context.sort(key=lambda r: r["context_length"], reverse=True)
        optimal_settings["context_length"] = successful_context[0]["context_length"]
    
    # Estimate batch size based on memory usage
    # This is a rough estimate - in real training there are gradients too
    if successful_context:
        memory_per_sample = {}
        for r in successful_context:
            if r["success"]:
                mem_key = "gpu_allocated_gb" if "gpu_allocated_gb" in r["memory_usage"] and r["memory_usage"]["gpu_allocated_gb"] > 0 else "ram_used_gb"
                memory_per_sample[r["context_length"]] = r["memory_usage"][mem_key]
        
        # If we have successful context length results
        if optimal_settings["context_length"] in memory_per_sample:
            # Get available memory
            mem_info = get_memory_usage()
            if torch.cuda.is_available():
                available_mem = mem_info["gpu_total_gb"] - mem_info["gpu_allocated_gb"]
                # Reserve 20% for overhead
                available_mem *= 0.8
            else:
                available_mem = mem_info["ram_total_gb"] * 0.5  # Use 50% of RAM
            
            # Estimate possible batch size
            sample_mem = memory_per_sample[optimal_settings["context_length"]]
            possible_batch_size = max(1, int(available_mem / (sample_mem * 3)))  # *3 for gradients and optimizer states
            
            optimal_settings["batch_size"] = possible_batch_size
            # Adjust gradient accumulation steps inversely with batch size
            optimal_settings["gradient_accumulation_steps"] = max(1, int(16 / possible_batch_size))
    
    return optimal_settings

def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen 2.5 Coder 14B")
    
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use for benchmark ('cpu', 'cuda', or 'auto')")
    parser.add_argument("--skip_16bit", action="store_true",
                        help="Skip 16-bit precision test")
    parser.add_argument("--skip_8bit", action="store_true",
                        help="Skip 8-bit quantization test")
    parser.add_argument("--skip_4bit", action="store_true",
                        help="Skip 4-bit quantization test")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log system information
    logger.info("=== System Information ===")
    sys_info = get_memory_usage()
    logger.info(f"CPU: {sys_info['cpu_count']} cores")
    logger.info(f"RAM: {sys_info['ram_total_gb']:.2f} GB")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {sys_info['gpu_name']}")
        logger.info(f"GPU Memory: {sys_info['gpu_total_gb']:.2f} GB")
    else:
        logger.info("No GPU detected.")
    
    # Run benchmarks
    results = {"model_loading": [], "context_length": []}
    
    # Test model loading with different quantization settings
    logger.info("\n=== Testing Model Loading ===")
    
    # 16-bit (standard)
    if not args.skip_16bit:
        result = test_model_loading(device=args.device)
        results["model_loading"].append(result)
    
    # 8-bit quantization
    if not args.skip_8bit:
        result = test_model_loading(device=args.device, use_8bit=True)
        results["model_loading"].append(result)
    
    # 4-bit quantization
    if not args.skip_4bit:
        result = test_model_loading(device=args.device, use_4bit=True) 
        results["model_loading"].append(result)
    
    # Determine best quantization method for context length tests
    best_quant = None
    
    for result in results["model_loading"]:
        if result["success"]:
            best_quant = result["quantization"]
            break
    
    if not best_quant:
        logger.error("Could not load model with any quantization method. Exiting.")
        return
    
    # Use best quantization method for context length tests
    logger.info(f"\n=== Testing Context Lengths with {best_quant} quantization ===")
    
    use_4bit = best_quant == "4bit"
    use_8bit = best_quant == "8bit"
    
    for length in CONTEXT_LENGTHS:
        result = test_context_length(
            context_length=length,
            device=args.device,
            use_4bit=use_4bit,
            use_8bit=use_8bit
        )
        results["context_length"].append(result)
        
        # Stop if we hit a context length that fails
        if not result["success"]:
            break
    
    # Save raw results
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    visualize_results(results, args.output_dir)
    
    # Determine optimal training settings
    optimal_settings = get_optimal_training_settings(results)
    
    logger.info("\n=== Optimal Training Settings ===")
    logger.info(f"Quantization: {optimal_settings['quantization']}")
    logger.info(f"Context Length: {optimal_settings['context_length']}")
    logger.info(f"Batch Size: {optimal_settings['batch_size']}")
    logger.info(f"Gradient Accumulation Steps: {optimal_settings['gradient_accumulation_steps']}")
    
    # Save optimal settings
    with open(os.path.join(args.output_dir, "optimal_settings.json"), "w") as f:
        json.dump(optimal_settings, f, indent=2)

if __name__ == "__main__":
    main() 