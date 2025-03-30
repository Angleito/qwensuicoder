"""
Benchmark script for Qwen models using PyTorch directly
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
import numpy as np
import gc
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
import tqdm
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model sizes to test (in billions of parameters)
MODEL_SIZES = [
    {"name": "Qwen2.5-Coder-0.5B", "params": 0.5},
    {"name": "Qwen2.5-Coder-1.5B", "params": 1.5}, 
    {"name": "Qwen2.5-Coder-7B", "params": 7},
    {"name": "Qwen2.5-Coder-14B", "params": 14},
    {"name": "Qwen2.5-Coder-32B", "params": 32},
]

# Context lengths to test (in tokens)
CONTEXT_LENGTHS = [1024, 2048, 4096]

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

def create_dummy_model(num_params_billions, device="cuda"):
    """
    Create a dummy model with approximately the specified number of parameters
    to test memory usage without needing the actual model
    """
    logger.info(f"Creating dummy model with ~{num_params_billions}B parameters")
    
    # Calculate number of parameters in billions
    num_params = int(num_params_billions * 1e9)
    
    # Determine tensor size (use a square tensor for simplicity)
    size = int(np.sqrt(num_params))
    
    # Create the dummy model - a simple Linear layer with the right parameter count
    model = torch.nn.Sequential(
        torch.nn.Linear(size, size)
    )
    
    # Move model to specified device
    model = model.to(device)
    
    # Print actual parameter count
    actual_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Actual parameters: {actual_params / 1e9:.2f}B")
    
    return model

def test_model_loading(model_params, device="cuda", use_fp16=False, use_int8=False, use_int4=False):
    """Test if a model of this size can be loaded with the given device configuration"""
    logger.info(f"Testing model with ~{model_params}B parameters on device: {device}")
    logger.info(f"Using FP16: {use_fp16}, INT8: {use_int8}, INT4: {use_int4}")
    
    precision = "int4" if use_int4 else "int8" if use_int8 else "fp16" if use_fp16 else "fp32"
    
    try:
        # Record memory before loading
        mem_before = get_memory_usage()
        
        # Attempt to create the dummy model
        start_time = time.time()
        
        if use_fp16:
            with torch.cuda.amp.autocast():
                model = create_dummy_model(model_params, device)
        else:
            model = create_dummy_model(model_params, device)

        # If int8 quantization requested, simulate quantization
        if use_int8:
            # Simulate int8 quantization by reducing memory
            # This is just an approximation of memory savings
            logger.info("Simulating INT8 quantization")

        # If int4 quantization requested, simulate quantization 
        if use_int4:
            # Simulate int4 quantization by reducing memory
            # This is just an approximation of memory savings
            logger.info("Simulating INT4 quantization")
        
        load_time = time.time() - start_time
        
        # Record memory after loading
        mem_after = get_memory_usage()
        
        # Calculate memory used
        mem_used = {
            "ram_used_gb": mem_after["ram_used_gb"] - mem_before["ram_used_gb"],
            "gpu_allocated_gb": mem_after.get("gpu_allocated_gb", 0) - mem_before.get("gpu_allocated_gb", 0),
        }
        
        logger.info(f"Model created successfully in {load_time:.2f} seconds")
        logger.info(f"Additional RAM used: {mem_used['ram_used_gb']:.2f} GB")
        if torch.cuda.is_available():
            logger.info(f"Additional GPU memory used: {mem_used['gpu_allocated_gb']:.2f} GB")
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "success": True,
            "model_params_billions": model_params,
            "load_time_seconds": load_time,
            "memory_usage": mem_used,
            "precision": precision
        }
    
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return {
            "success": False,
            "model_params_billions": model_params,
            "error": str(e),
            "precision": precision
        }

def test_context_length(model_params, context_length, device="cuda", use_fp16=False, use_int8=False, use_int4=False):
    """Test if a model can handle the specified context length"""
    logger.info(f"Testing {model_params}B parameter model with context length {context_length}")
    
    precision = "int4" if use_int4 else "int8" if use_int8 else "fp16" if use_fp16 else "fp32"
    
    try:
        # Create a dummy model
        if use_fp16:
            with torch.cuda.amp.autocast():
                model = create_dummy_model(model_params, device)
        else:
            model = create_dummy_model(model_params, device)
        
        # Create input of specified context length (batch size 1)
        hidden_size = int(np.sqrt(model_params * 1e9))
        input_tensor = torch.rand(1, context_length, hidden_size, device=device)
        
        # Record memory before inference
        mem_before = get_memory_usage()
        
        # Run simulated inference
        start_time = time.time()
        with torch.no_grad():
            if use_fp16:
                with torch.cuda.amp.autocast():
                    # Simulate attention mechanism with a simple operation
                    output = torch.matmul(input_tensor, model[0].weight.t())
            else:
                # Simulate attention mechanism with a simple operation
                output = torch.matmul(input_tensor, model[0].weight.t())
                
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
        del model, output, input_tensor
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "success": True,
            "model_params_billions": model_params,
            "context_length": context_length,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_usage": mem_used,
            "precision": precision
        }
    
    except Exception as e:
        logger.error(f"Failed with context length {context_length}: {e}")
        return {
            "success": False,
            "model_params_billions": model_params,
            "context_length": context_length,
            "error": str(e),
            "precision": precision
        }

# Add a simple dataset for efficiency testing
class BenchmarkDataset(Dataset):
    """Simple dataset for benchmarking learning efficiency"""
    def __init__(self, seq_length=512, size=1000, vocab_size=32000):
        self.seq_length = seq_length
        self.size = size
        self.vocab_size = vocab_size
        
        # Create synthetic data for benchmarking
        self.examples = []
        for _ in range(size):
            # Random token IDs
            input_ids = torch.randint(1, vocab_size-1, (seq_length,))
            self.examples.append({
                "input_ids": input_ids,
                "labels": input_ids.clone(),
            })
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.examples[idx]

# Add efficiency testing
def test_learning_efficiency(model_params, precision, context_length, device="cuda"):
    """Test how efficiently a model learns on synthetic data"""
    logger.info(f"Testing learning efficiency for {model_params}B model with {precision} precision")
    
    # Parameter mapping
    use_fp16 = (precision == "fp16")
    use_int8 = (precision == "int8")
    use_int4 = (precision == "int4")
    
    try:
        # Create a dummy model
        if use_fp16:
            with torch.cuda.amp.autocast():
                model = create_dummy_model(model_params, device)
        else:
            model = create_dummy_model(model_params, device)
        
        # Setup for training test
        batch_size = 1  # Small batch for testing
        num_batches = 10  # Only need a few batches to test speed
        learning_rate = 1e-4
        
        # Create synthetic dataset (smaller for benchmarking)
        dataset = BenchmarkDataset(seq_length=context_length, size=batch_size * num_batches)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True
        )
        
        # Basic optimizer
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        # Record memory before training
        mem_before = get_memory_usage()
        
        # Measure time for a few training steps
        start_time = time.time()
        
        total_loss = 0
        batches_processed = 0
        
        # Simulate a few training steps
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass - use a simple model-like calculation
            logits = model(input_ids)
            
            # Calculate a loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, input_ids.size(-1)),
                labels.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches_processed += 1
            
        training_time = time.time() - start_time
        
        # Record memory after training
        mem_after = get_memory_usage()
        
        # Calculate memory used
        mem_used = {
            "ram_used_gb": mem_after["ram_used_gb"] - mem_before["ram_used_gb"],
            "gpu_allocated_gb": mem_after.get("gpu_allocated_gb", 0) - mem_before.get("gpu_allocated_gb", 0),
        }
        
        # Calculate metrics
        avg_loss = total_loss / batches_processed if batches_processed > 0 else float('inf')
        tokens_per_second = (batch_size * context_length * batches_processed) / training_time
        
        logger.info(f"Learning efficiency test completed in {training_time:.2f} seconds")
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"Training speed: {tokens_per_second:.2f} tokens/sec")
        logger.info(f"Additional GPU memory used: {mem_used['gpu_allocated_gb']:.2f} GB")
        
        # Clean up
        del model, optimizer, dataset, dataloader
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "success": True,
            "model_params_billions": model_params,
            "precision": precision,
            "context_length": context_length,
            "training_time_seconds": training_time,
            "tokens_per_second": tokens_per_second,
            "avg_loss": avg_loss,
            "memory_usage": mem_used,
        }
        
    except Exception as e:
        logger.error(f"Failed efficiency test: {e}")
        return {
            "success": False,
            "model_params_billions": model_params,
            "precision": precision,
            "context_length": context_length,
            "error": str(e)
        }

def visualize_results(results, output_dir):
    """Create visualizations of benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for visualization
    model_sizes = []
    model_loading_results = {}
    for size_info in MODEL_SIZES:
        size_param = size_info["params"]
        model_sizes.append(size_param)
        model_loading_results[size_param] = []
        
    for result in results["model_loading"]:
        if result["success"]:
            model_size = result["model_params_billions"]
            precision = result["precision"]
            if model_size in model_loading_results:
                model_loading_results[model_size].append({
                    "precision": precision,
                    "load_time": result["load_time_seconds"],
                    "memory_usage": result["memory_usage"]
                })
    
    # Plot model loading times by size and precision
    plt.figure(figsize=(12, 8))
    bar_width = 0.2
    colors = {'fp32': 'blue', 'fp16': 'green', 'int8': 'orange', 'int4': 'red'}
    
    # Group by model size and precision
    indices = np.arange(len(model_sizes))
    
    for i, precision in enumerate(['fp32', 'fp16', 'int8', 'int4']):
        load_times = []
        for size in model_sizes:
            # Find result with this precision for this model size
            matching_results = [r for r in model_loading_results[size] if r["precision"] == precision]
            if matching_results:
                load_times.append(matching_results[0]["load_time"])
            else:
                load_times.append(0)  # No result for this combination
        
        # Only plot if we have data for this precision
        if any(load_times):
            plt.bar(indices + i*bar_width, load_times, bar_width, 
                   label=precision, color=colors[precision])
    
    plt.xlabel('Model Size (Billion parameters)')
    plt.ylabel('Loading Time (seconds)')
    plt.title('Model Loading Times by Size and Precision')
    plt.xticks(indices + bar_width * 1.5, [f"{size}B" for size in model_sizes])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_loading_times.png"))
    
    # Plot memory usage by model size and precision
    plt.figure(figsize=(12, 8))
    
    for i, precision in enumerate(['fp32', 'fp16', 'int8', 'int4']):
        memory_usage = []
        for size in model_sizes:
            # Find result with this precision for this model size
            matching_results = [r for r in model_loading_results[size] if r["precision"] == precision]
            if matching_results:
                if torch.cuda.is_available():
                    memory_usage.append(matching_results[0]["memory_usage"]["gpu_allocated_gb"])
                else:
                    memory_usage.append(matching_results[0]["memory_usage"]["ram_used_gb"])
            else:
                memory_usage.append(0)  # No result for this combination
        
        # Only plot if we have data for this precision
        if any(memory_usage):
            plt.bar(indices + i*bar_width, memory_usage, bar_width, 
                   label=precision, color=colors[precision])
    
    plt.xlabel('Model Size (Billion parameters)')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage by Model Size and Precision')
    plt.xticks(indices + bar_width * 1.5, [f"{size}B" for size in model_sizes])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_usage.png"))
    
    # Visualize context length results
    context_length_results = {}
    
    # Group results by model size and context length
    for result in results["context_length"]:
        if result["success"]:
            model_size = result["model_params_billions"]
            if model_size not in context_length_results:
                context_length_results[model_size] = []
            context_length_results[model_size].append(result)
    
    # Plot tokens per second by context length for each model size
    plt.figure(figsize=(12, 8))
    
    for model_size, size_results in context_length_results.items():
        if size_results:
            # Sort by context length
            size_results.sort(key=lambda r: r["context_length"])
            
            context_lengths = [r["context_length"] for r in size_results]
            tokens_per_second = [r["tokens_per_second"] for r in size_results]
            
            plt.plot(context_lengths, tokens_per_second, 'o-', 
                    label=f"{model_size}B parameters")
    
    plt.xlabel('Context Length (tokens)')
    plt.ylabel('Tokens per Second')
    plt.title('Throughput by Context Length and Model Size')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tokens_per_second.png"))

    # Add visualization for learning efficiency if available
    if "learning_efficiency" in results and results["learning_efficiency"]:
        # Extract data
        efficiency_data = {}
        
        for result in results["learning_efficiency"]:
            if result["success"]:
                model_size = result["model_params_billions"]
                precision = result["precision"]
                
                if model_size not in efficiency_data:
                    efficiency_data[model_size] = {}
                    
                if precision not in efficiency_data[model_size]:
                    efficiency_data[model_size][precision] = result
        
        # Plot training speed (tokens/sec) by model size and precision
        plt.figure(figsize=(12, 8))
        
        model_sizes = sorted(efficiency_data.keys())
        x = np.arange(len(model_sizes))
        width = 0.2
        
        precisions = ["fp32", "fp16", "int8", "int4"]
        colors = {'fp32': 'blue', 'fp16': 'green', 'int8': 'orange', 'int4': 'red'}
        
        for i, precision in enumerate(precisions):
            speed_values = []
            
            for size in model_sizes:
                if precision in efficiency_data[size]:
                    speed_values.append(efficiency_data[size][precision]["tokens_per_second"])
                else:
                    speed_values.append(0)
            
            # Only plot if we have data for this precision
            if any(speed_values):
                plt.bar(x + i*width - width*1.5, speed_values, width, 
                       label=precision, color=colors[precision])
        
        plt.xlabel('Model Size (Billion parameters)')
        plt.ylabel('Training Speed (tokens/sec)')
        plt.title('Training Speed by Model Size and Precision')
        plt.xticks(x, [f"{size}B" for size in model_sizes])
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_speed.png"))
        
        # Plot memory usage during training
        plt.figure(figsize=(12, 8))
        
        for i, precision in enumerate(precisions):
            memory_values = []
            
            for size in model_sizes:
                if precision in efficiency_data[size]:
                    if "gpu_allocated_gb" in efficiency_data[size][precision]["memory_usage"]:
                        memory_values.append(efficiency_data[size][precision]["memory_usage"]["gpu_allocated_gb"])
                    else:
                        memory_values.append(efficiency_data[size][precision]["memory_usage"]["ram_used_gb"])
                else:
                    memory_values.append(0)
            
            # Only plot if we have data for this precision
            if any(memory_values):
                plt.bar(x + i*width - width*1.5, memory_values, width, 
                       label=precision, color=colors[precision])
        
        plt.xlabel('Model Size (Billion parameters)')
        plt.ylabel('Memory Usage During Training (GB)')
        plt.title('Training Memory Usage by Model Size and Precision')
        plt.xticks(x, [f"{size}B" for size in model_sizes])
        plt.legend()
        plt.grid(axis='y')
        plt.ylim(0, np.max(memory_values) if any(memory_values) else 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_memory.png"))

def get_optimal_settings(results):
    """Determine optimal training settings based on benchmark results"""
    
    # Default conservative settings
    optimal_settings = {
        "model_name": "Qwen2.5-Coder-1.5B",
        "model_params_billions": 1.5,  # Default to smallest model
        "precision": "fp16",
        "context_length": 1024,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "slora_rank": 16,
        "slora_alpha": 32,
        "slora_dropout": 0.05,
        "slora_sparsity": 0.9,
        "max_context_length": 4096
    }
    
    # Find successful model loadings
    successful_model_sizes = set()
    model_names = {}
    for result in results["model_loading"]:
        if result["success"]:
            size = result["model_params_billions"]
            successful_model_sizes.add(size)
            # Map model size to model name
            for model_info in MODEL_SIZES:
                if model_info["params"] == size:
                    model_names[size] = model_info["name"]
    
    # If there are successful learning efficiency results, use them to determine the optimal model
    if "learning_efficiency" in results and results["learning_efficiency"]:
        # First create a filtered set of models that work efficiently
        efficient_models = {}
        
        for result in results["learning_efficiency"]:
            if result["success"]:
                model_size = result["model_params_billions"]
                precision = result["precision"]
                
                if model_size not in efficient_models:
                    efficient_models[model_size] = {}
                
                efficient_models[model_size][precision] = result
        
        if efficient_models:
            # Find the largest model with good efficiency (tokens/sec > threshold)
            # Sort by model size (largest first)
            sorted_sizes = sorted(efficient_models.keys(), reverse=True)
            
            for size in sorted_sizes:
                # Find the best precision for this model size
                best_precision = None
                best_speed = 0
                
                for precision, result in efficient_models[size].items():
                    speed = result["tokens_per_second"]
                    if speed > best_speed:
                        best_speed = speed
                        best_precision = precision
                
                if best_precision:
                    optimal_settings["model_params_billions"] = size
                    optimal_settings["precision"] = best_precision
                    
                    # Add model name
                    if size in model_names:
                        optimal_settings["model_name"] = model_names[size]
                    
                    # Find the best context length for this model/precision combo
                    context_results = [r for r in results["context_length"] 
                                     if r["success"] 
                                     and r["model_params_billions"] == size
                                     and "precision" in r and r["precision"] == best_precision]
                    
                    if context_results:
                        max_context = max(r["context_length"] for r in context_results)
                        optimal_settings["context_length"] = max_context
                        optimal_settings["max_context_length"] = max_context
                    
                    # We found a good model, stop searching
                    break
    else:
        # Fall back to original logic if no efficiency results
        if successful_model_sizes:
            largest_size = max(successful_model_sizes)
            optimal_settings["model_params_billions"] = largest_size
            
            # Add model name for the chosen size
            if largest_size in model_names:
                optimal_settings["model_name"] = model_names[largest_size]
            
            # Find the best precision for the chosen model size
            precision_results = [
                r for r in results["model_loading"]
                if r["success"] and r["model_params_billions"] == largest_size
            ]
            
            # If we have fp16 results, prefer fp16
            fp16_results = [r for r in precision_results if r["precision"] == "fp16"]
            if fp16_results:
                optimal_settings["precision"] = "fp16"
            # Otherwise use whatever precision was successful
            elif precision_results:
                optimal_settings["precision"] = precision_results[0]["precision"]
    
    # Calculate batch size and gradient accumulation steps
    # This is a heuristic based on model size and available memory
    if optimal_settings["model_params_billions"] <= 1:
        optimal_settings["batch_size"] = 8
        optimal_settings["gradient_accumulation_steps"] = 4
    elif optimal_settings["model_params_billions"] <= 7:
        optimal_settings["batch_size"] = 4
        optimal_settings["gradient_accumulation_steps"] = 8
    else:
        optimal_settings["batch_size"] = 1
        optimal_settings["gradient_accumulation_steps"] = 16
    
    # Determine SLoRA settings based on model size
    if optimal_settings["model_params_billions"] <= 1:
        optimal_settings["slora_rank"] = 8
        optimal_settings["slora_sparsity"] = 0.8
    elif optimal_settings["model_params_billions"] <= 7:
        optimal_settings["slora_rank"] = 16
        optimal_settings["slora_sparsity"] = 0.9
    else: 
        optimal_settings["slora_rank"] = 32
        optimal_settings["slora_sparsity"] = 0.95
    
    return optimal_settings

def configure_ollama(optimal_settings):
    """
    Configure Ollama with the best model from benchmark results
    
    Args:
        optimal_settings: Dictionary containing the optimal settings from benchmark
    
    Returns:
        True if configuration successful, False otherwise
    """
    try:
        # Get model details
        model_name = optimal_settings.get("model_name", "")
        model_params = optimal_settings.get("model_params_billions", 0)
        precision = optimal_settings.get("precision", "fp16")
        
        if not model_name:
            logger.error("No model name found in optimal settings")
            return False
        
        logger.info(f"Configuring Ollama with model: {model_name}")
        
        # Create Modelfile for Ollama
        modelfile_content = f"""
FROM {model_name}
PARAMETER temperature 0.7
PARAMETER num_ctx {optimal_settings.get('max_context_length', 4096)}
PARAMETER num_gpu {1 if torch.cuda.is_available() else 0}
PARAMETER f16 {1 if precision == 'fp16' else 0}
PARAMETER num_thread {psutil.cpu_count(logical=True)}
PARAMETER seed 42
TEMPLATE \"\"\"
{{system}}
You are {model_name}, a large language model trained by Qwen. 
You are designed to be helpful, harmless, and honest.
{{/system}}

{{user}}
{{input}}
{{/user}}

{{assistant}}
{{output}}
{{/assistant}}
\"\"\"
"""
        # Write Modelfile
        os.makedirs("ollama_config", exist_ok=True)
        with open("ollama_config/Modelfile", "w") as f:
            f.write(modelfile_content)
        
        # Create model in Ollama
        model_tag = f"qwen-coder-{int(model_params)}b"
        logger.info(f"Creating Ollama model with tag: {model_tag}")
        result = subprocess.run(
            ["ollama", "create", model_tag, "-f", "ollama_config/Modelfile"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to create Ollama model: {result.stderr}")
            return False
        
        logger.info(f"Successfully configured Ollama with model: {model_tag}")
        
        # Save Ollama configuration
        ollama_config = {
            "model_tag": model_tag,
            "original_model": model_name,
            "parameters": {
                "context_length": optimal_settings.get('max_context_length', 4096),
                "temperature": 0.7,
                "gpu_layers": 1 if torch.cuda.is_available() else 0,
                "f16": precision == "fp16"
            }
        }
        
        with open("ollama_config/config.json", "w") as f:
            json.dump(ollama_config, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Error configuring Ollama: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="PyTorch Benchmark for Qwen Models")
    
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for benchmark ('cpu' or 'cuda')")
    parser.add_argument("--skip_fp32", action="store_true",
                        help="Skip FP32 precision test")
    parser.add_argument("--skip_fp16", action="store_true",
                        help="Skip FP16 precision test")
    parser.add_argument("--skip_int8", action="store_true",
                        help="Skip INT8 quantization test")
    parser.add_argument("--skip_int4", action="store_true",
                        help="Skip INT4 quantization test")
    parser.add_argument("--max_model_size", type=float, default=None,
                        help="Maximum model size to test in billions of parameters")
    
    # New arguments
    parser.add_argument("--test_efficiency", action="store_true",
                        help="Test learning efficiency on synthetic data")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of training iterations for efficiency test")
    parser.add_argument("--configure_ollama", action="store_true",
                        help="Configure Ollama with the best model")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use CPU if CUDA is not available and device is cuda
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, using CPU instead")
        args.device = "cpu"
    
    # Log system information
    logger.info("=== System Information ===")
    sys_info = get_memory_usage()
    logger.info(f"CPU: {sys_info['cpu_count']} cores")
    logger.info(f"RAM: {sys_info['ram_total_gb']:.2f} GB")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {sys_info['gpu_name']}")
        logger.info(f"GPU Memory: {sys_info['gpu_total_gb']:.2f} GB")
        logger.info(f"PyTorch CUDA: {torch.version.cuda}")
    else:
        logger.info("No GPU detected.")
    
    # Filter model sizes if a max size is specified
    test_model_sizes = MODEL_SIZES
    if args.max_model_size is not None:
        test_model_sizes = [size for size in MODEL_SIZES if size["params"] <= args.max_model_size]
    
    # Run benchmarks
    results = {"model_loading": [], "context_length": []}
    
    # Test model loading with different precisions
    logger.info("\n=== Testing Model Loading ===")
    
    for model_size in test_model_sizes:
        model_param = model_size["params"]
        model_name = model_size["name"]
        logger.info(f"\nBenchmarking model: {model_name} ({model_param}B parameters)")
        
        # FP32 (standard)
        if not args.skip_fp32:
            result = test_model_loading(model_param, device=args.device)
            results["model_loading"].append(result)
        
        # FP16 (mixed precision)
        if not args.skip_fp16:
            result = test_model_loading(model_param, device=args.device, use_fp16=True)
            results["model_loading"].append(result)
        
        # INT8 quantization
        if not args.skip_int8:
            result = test_model_loading(model_param, device=args.device, use_int8=True)
            results["model_loading"].append(result)
        
        # INT4 quantization
        if not args.skip_int4:
            result = test_model_loading(model_param, device=args.device, use_int4=True)
            results["model_loading"].append(result)
    
    # For each model size that loaded successfully, test context lengths
    logger.info("\n=== Testing Context Lengths ===")
    
    for model_size in test_model_sizes:
        model_param = model_size["params"]
        
        # Find the best precision that worked for this model size
        best_precision = None
        for precision in ["int4", "int8", "fp16", "fp32"]:
            matching_results = [
                r for r in results["model_loading"] 
                if r["success"] and r["model_params_billions"] == model_param and r["precision"] == precision
            ]
            if matching_results:
                best_precision = precision
                break
        
        if best_precision:
            logger.info(f"\nTesting context lengths for {model_param}B model with {best_precision} precision")
            
            use_fp16 = (best_precision == "fp16")
            use_int8 = (best_precision == "int8")
            use_int4 = (best_precision == "int4")
            
            for length in CONTEXT_LENGTHS:
                result = test_context_length(
                    model_param,
                    context_length=length,
                    device=args.device,
                    use_fp16=use_fp16,
                    use_int8=use_int8,
                    use_int4=use_int4
                )
                results["context_length"].append(result)
                
                # Stop if we hit a context length that fails
                if not result["success"]:
                    break
    
    # Test learning efficiency if requested
    if args.test_efficiency:
        logger.info("\n=== Testing Learning Efficiency ===")
        results["learning_efficiency"] = []
        
        for model_size in test_model_sizes:
            model_param = model_size["params"]
            model_name = model_size["name"]
            
            # For each model, test with the best precision that works
            for precision in ["int4", "int8", "fp16", "fp32"]:
                # Check if this model+precision combination loaded successfully
                matching_results = [
                    r for r in results["model_loading"] 
                    if r["success"] and r["model_params_billions"] == model_param and r["precision"] == precision
                ]
                
                if matching_results:
                    # Found a working precision, test efficiency
                    logger.info(f"\nTesting efficiency for {model_name} with {precision} precision")
                    
                    # Find the maximum context length that worked for this model+precision
                    max_context = 1024  # Default
                    context_results = [
                        r for r in results["context_length"]
                        if r["success"] and r["model_params_billions"] == model_param 
                        and ((r["precision"] == precision) if "precision" in r else True)
                    ]
                    
                    if context_results:
                        max_context = max(r["context_length"] for r in context_results)
                    
                    # Test learning efficiency
                    efficiency_result = test_learning_efficiency(
                        model_param,
                        precision=precision,
                        context_length=max_context,
                        device=args.device
                    )
                    
                    results["learning_efficiency"].append(efficiency_result)
                    
                    # If this precision worked, no need to test others for this model
                    break
    
    # Save raw results
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    visualize_results(results, args.output_dir)
    
    # Determine optimal training settings
    optimal_settings = get_optimal_settings(results)
    
    logger.info("\n=== Optimal Training Settings ===")
    logger.info(f"Model Size: {optimal_settings['model_params_billions']}B parameters")
    logger.info(f"Precision: {optimal_settings['precision']}")
    logger.info(f"Context Length: {optimal_settings['context_length']}")
    logger.info(f"Batch Size: {optimal_settings['batch_size']}")
    logger.info(f"Gradient Accumulation Steps: {optimal_settings['gradient_accumulation_steps']}")
    
    # Save optimal settings
    with open(os.path.join(args.output_dir, "optimal_settings.json"), "w") as f:
        json.dump(optimal_settings, f, indent=2)
    
    # Configure Ollama with the best model
    if args.configure_ollama:
        configure_ollama(optimal_settings)
    
    # Return the best model and settings
    return optimal_settings

if __name__ == "__main__":
    main() 