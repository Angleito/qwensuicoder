#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Manager for Qwen2.5-Coder models
Provides a unified interface for PyTorch, Ollama, and SLora
"""

import os
import json
import logging
import subprocess
import torch
import argparse
import requests
from typing import Dict, Any, Optional, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenCoderManager:
    """
    Manager for Qwen2.5-Coder models that handles:
    - Loading best model from benchmark
    - Configuring Ollama
    - Training with SLora
    - Inference with PyTorch directly
    """
    
    def __init__(self, benchmark_dir="benchmark_results", 
                 models_dir="trained_models", ollama_dir="ollama_config"):
        self.benchmark_dir = benchmark_dir
        self.models_dir = models_dir
        self.ollama_dir = ollama_dir
        
        # Create directories if they don't exist
        os.makedirs(benchmark_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(ollama_dir, exist_ok=True)
        
        # Load settings if available
        self.optimal_settings = self._load_optimal_settings()
        self.slora_config = self._load_slora_config()
        self.ollama_config = self._load_ollama_config()
    
    def _load_optimal_settings(self) -> Dict[str, Any]:
        """Load optimal settings from benchmark results"""
        settings_path = os.path.join(self.benchmark_dir, "optimal_settings.json")
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading optimal settings: {e}")
        return {}
    
    def _load_slora_config(self) -> Dict[str, Any]:
        """Load SLora configuration if available"""
        slora_path = os.path.join(self.models_dir, "slora_config.json")
        if os.path.exists(slora_path):
            try:
                with open(slora_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading SLora config: {e}")
        return {}
    
    def _load_ollama_config(self) -> Dict[str, Any]:
        """Load Ollama configuration if available"""
        ollama_path = os.path.join(self.ollama_dir, "config.json")
        if os.path.exists(ollama_path):
            try:
                with open(ollama_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading Ollama config: {e}")
        return {}
    
    def run_benchmark(self, max_model_size=1.5, configure_ollama=True) -> Dict[str, Any]:
        """
        Run benchmark to find the optimal model and settings
        
        Args:
            max_model_size: Maximum model size to test in billions of parameters (default: 1.5B)
            configure_ollama: Whether to configure Ollama with the best model
            
        Returns:
            Dictionary of optimal settings
        """
        logger.info(f"Running benchmark with max model size: {max_model_size}B")
        
        # Build command
        cmd = [
            "python", "benchmark_qwen_pytorch.py",
            "--output_dir", self.benchmark_dir,
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            "--max_model_size", str(max_model_size),
            "--test_efficiency"
        ]
        
        if configure_ollama:
            cmd.append("--configure_ollama")
        
        # Run benchmark
        try:
            subprocess.run(cmd, check=True)
            
            # Reload settings
            self.optimal_settings = self._load_optimal_settings()
            self.ollama_config = self._load_ollama_config()
            
            return self.optimal_settings
        except subprocess.CalledProcessError as e:
            logger.error(f"Benchmark failed: {e}")
            return {}
    
    def configure_ollama(self, force=False) -> bool:
        """
        Configure Ollama with the best model from benchmark
        
        Args:
            force: Whether to force reconfiguration even if already configured
            
        Returns:
            True if successful, False otherwise
        """
        if not self.optimal_settings:
            logger.error("No optimal settings available. Run benchmark first.")
            return False
        
        if self.ollama_config and not force:
            logger.info("Ollama already configured. Use force=True to reconfigure.")
            return True
        
        # Check if Ollama is installed
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Ollama not found or not working properly")
                return False
        except Exception:
            logger.error("Ollama not installed or not in PATH")
            return False
        
        # Configure Ollama
        from benchmark_qwen_pytorch import configure_ollama
        return configure_ollama(self.optimal_settings)
    
    def train_model(self, output_dir=None, epochs=3, learning_rate=2e-4) -> bool:
        """
        Train model with SLora using optimal settings
        
        Args:
            output_dir: Directory to save trained model, defaults to models_dir/model_name
            epochs: Number of epochs to train for
            learning_rate: Learning rate for training
            
        Returns:
            True if successful, False otherwise
        """
        if not self.optimal_settings:
            logger.error("No optimal settings available. Run benchmark first.")
            return False
        
        # Set output directory
        if output_dir is None:
            model_name = self.optimal_settings.get("model_name", "qwen-coder")
            model_name = model_name.lower().replace(".", "-")
            output_dir = os.path.join(self.models_dir, model_name)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Build command
        cmd = [
            "python", "train_qwen_pytorch.py",
            "--settings_file", os.path.join(self.benchmark_dir, "optimal_settings.json"),
            "--output_dir", output_dir,
            "--epochs", str(epochs),
            "--lr", str(learning_rate),
            "--fp16" if self.optimal_settings.get("precision") == "fp16" else "--no-fp16",
            "--sparsity", str(self.optimal_settings.get("slora_sparsity", 0.95)),
            "--rank", str(self.optimal_settings.get("slora_rank", 16)),
            "--adaptive_sparsity",
            "--full_resource_utilization"
        ]
        
        # Run training
        try:
            logger.info(f"Starting training with command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Save SLora config
            slora_config = {
                "model_name": self.optimal_settings.get("model_name", ""),
                "rank": self.optimal_settings.get("slora_rank", 16),
                "sparsity": self.optimal_settings.get("slora_sparsity", 0.95),
                "output_dir": output_dir
            }
            
            with open(os.path.join(self.models_dir, "slora_config.json"), "w") as f:
                json.dump(slora_config, f, indent=2)
            
            # Reload config
            self.slora_config = self._load_slora_config()
            
            logger.info(f"Training completed successfully. Model saved to {output_dir}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with error code {e.returncode}: {e}")
            return False
        except Exception as e:
            logger.error(f"Training failed with exception: {e}")
            return False
    
    def inference_with_ollama(self, prompt, model_tag=None) -> str:
        """
        Run inference with Ollama
        
        Args:
            prompt: Prompt to generate response for
            model_tag: Custom model tag, defaults to best model from benchmark
            
        Returns:
            Generated response
        """
        if model_tag is None:
            if not self.ollama_config:
                logger.error("No Ollama configuration available. Configure Ollama first.")
                return ""
            model_tag = self.ollama_config.get("model_tag", "")
        
        if not model_tag:
            logger.error("No model tag specified")
            return ""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_tag, "prompt": prompt, "stream": False}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama inference failed: {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return ""
    
    def inference_with_pytorch(self, prompt, use_slora=True) -> str:
        """
        Run inference directly with PyTorch
        
        Args:
            prompt: Prompt to generate response for
            use_slora: Whether to use SLora weights
            
        Returns:
            Generated response
        """
        if not self.optimal_settings:
            logger.error("No optimal settings available. Run benchmark first.")
            return ""
        
        # Import here to avoid circular imports
        from train_qwen_pytorch import load_model_from_benchmark, SimpleTokenizer
        
        try:
            # Load model
            model = load_model_from_benchmark(
                os.path.join(self.benchmark_dir, "optimal_settings.json")
            )
            
            # Apply SLora weights if requested
            if use_slora and self.slora_config:
                slora_path = os.path.join(
                    self.slora_config.get("output_dir", ""), 
                    "slora_weights_final.pt"
                )
                
                if os.path.exists(slora_path):
                    logger.info(f"Loading SLora weights from {slora_path}")
                    slora_weights = torch.load(slora_path)
                    
                    # Apply weights to model
                    for name, module in model.named_modules():
                        if hasattr(module, "lora_A") and f"{name}.lora_A" in slora_weights:
                            module.lora_A.data = slora_weights[f"{name}.lora_A"]
                            module.lora_B.data = slora_weights[f"{name}.lora_B"]
                            
                            if f"{name}.sparsity_mask" in slora_weights:
                                module.sparsity_mask.data = slora_weights[f"{name}.sparsity_mask"]
            
            # Create tokenizer
            tokenizer = SimpleTokenizer()
            
            # Create input prompt (simple version without chat formatting)
            input_text = prompt
            
            # Encode prompt
            input_ids = tokenizer.encode(input_text)
            input_tensor = torch.tensor([input_ids], device=model.device)
            
            # Generate response
            logger.info(f"Generating response for prompt: {prompt[:50]}...")
            max_length = min(1024, self.optimal_settings.get("max_context_length", 1024))
            
            with torch.no_grad():
                if self.optimal_settings.get("precision") == "fp16":
                    with torch.cuda.amp.autocast():
                        outputs = model.forward(
                            input_ids=input_tensor,
                            attention_mask=torch.ones_like(input_tensor)
                        )
                        
                        # Get logits and sample from them
                        logits = outputs.logits[:, -1, :]
                        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                        
                        # Simple greedy generation
                        generated_ids = [input_ids]
                        for _ in range(max_length):
                            # Forward pass with generated tokens
                            outputs = model.forward(
                                input_ids=next_token_id,
                                attention_mask=torch.ones_like(next_token_id)
                            )
                            logits = outputs.logits[:, -1, :]
                            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                            
                            # Stop if we predict end of sequence
                            if next_token_id.item() == tokenizer.eos_token_id:
                                break
                                
                            generated_ids.append(next_token_id.item())
                else:
                    outputs = model.forward(
                        input_ids=input_tensor,
                        attention_mask=torch.ones_like(input_tensor)
                    )
                    
                    # Get logits and sample from them
                    logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                    
                    # Simple greedy generation
                    generated_ids = [input_ids]
                    for _ in range(max_length):
                        # Forward pass with generated tokens
                        outputs = model.forward(
                            input_ids=next_token_id,
                            attention_mask=torch.ones_like(next_token_id)
                        )
                        logits = outputs.logits[:, -1, :]
                        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                        
                        # Stop if we predict end of sequence
                        if next_token_id.item() == tokenizer.eos_token_id:
                            break
                            
                        generated_ids.append(next_token_id.item())
            
            # Decode output
            response = tokenizer.decode(generated_ids)
            
            # Remove the prompt from the response
            response_only = response.replace(input_text, "").strip()
            
            return response_only
        except Exception as e:
            logger.error(f"PyTorch inference error: {e}")
            logger.exception("Detailed traceback:")
            return f"Error generating response: {str(e)}"

def main():
    """Command line interface for model manager"""
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B Model Manager")
    parser.add_argument("--action", type=str, required=True, 
                        choices=["benchmark", "train", "configure-ollama", "infer-ollama", "infer-pytorch"],
                        help="Action to perform")
    parser.add_argument("--max-model-size", type=float, default=1.5, 
                        help="Maximum model size in billions of parameters (default: 1.5)")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of epochs for training")
    parser.add_argument("--learning-rate", type=float, default=2e-4, 
                        help="Learning rate for training")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Output directory for trained model")
    parser.add_argument("--prompt", type=str, default="", 
                        help="Prompt for inference")
    parser.add_argument("--force", action="store_true", 
                        help="Force reconfiguration of Ollama")
    parser.add_argument("--no-slora", action="store_true",
                        help="Don't use SLora for inference")
    args = parser.parse_args()
    
    # Create manager
    manager = QwenCoderManager()
    
    # Perform action
    if args.action == "benchmark":
        result = manager.run_benchmark(max_model_size=args.max_model_size)
        if result:
            print(f"Benchmark completed. Best model: {result.get('model_name', 'Unknown')}")
            print(f"Precision: {result.get('precision', 'Unknown')}")
            print(f"Max context length: {result.get('max_context_length', 'Unknown')}")
    
    elif args.action == "train":
        success = manager.train_model(
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        if success:
            print("Training completed successfully")
    
    elif args.action == "infer-ollama":
        response = manager.inference_with_ollama(args.prompt)
        print(f"Response: {response}")
    
    elif args.action == "infer-pytorch":
        response = manager.inference_with_pytorch(args.prompt, use_slora=not args.no_slora)
        print(f"Response: {response}")
    
    elif args.action == "configure-ollama":
        success = manager.configure_ollama(force=args.force)
        if success:
            print("Ollama configured successfully")

if __name__ == "__main__":
    main() 