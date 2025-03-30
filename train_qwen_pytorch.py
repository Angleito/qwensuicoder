"""
Train Qwen model with PyTorch and SLoRA (Sparse Low-Rank Adaptation)
Optimized for local usage without Hugging Face dependencies
"""

import os
import torch
import argparse
import json
import logging
import numpy as np
import time
import math
import random
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.cuda.amp import autocast, GradScaler
import psutil
import gc
import tqdm
from safetensors.torch import save_file
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SparseLinearLayer(torch.nn.Module):
    """
    Linear layer with sparse low-rank adaptation
    """
    def __init__(self, base_layer, rank=16, alpha=32, dropout=0.05, sparsity=0.9):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.sparsity = sparsity
        
        # Get dimensions
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # Initialize LoRA matrices (A: in_features x rank, B: rank x out_features)
        self.lora_A = torch.nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Initialize A with Kaiming uniform
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Initialize B with zeros
        torch.nn.init.zeros_(self.lora_B)
        
        # Create dropout layer
        self.lora_dropout = torch.nn.Dropout(p=dropout)
        
        # Create sparsity mask
        self.sparsity_mask = self.create_sparsity_mask(self.lora_A.shape)
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def create_sparsity_mask(self, shape):
        """Create a binary mask for sparsity"""
        # Create random binary mask with specified sparsity
        mask = torch.rand(shape, device=self.lora_A.device) > self.sparsity
        return mask
    
    def forward(self, x):
        # Original forward pass
        base_output = self.base_layer(x)
        
        # Apply SLoRA
        x_dropout = self.lora_dropout(x)
        
        # Apply sparsity mask (element-wise multiply)
        sparse_A = self.lora_A * self.sparsity_mask
        
        # LoRA forward pass: x → xA → xAB → scaled_xAB
        lora_output = (x_dropout @ sparse_A) @ self.lora_B
        
        # Scale output
        return base_output + (lora_output * self.scaling)
    
    def get_sparsity_stats(self):
        """Returns the actual sparsity percentage"""
        active_params = torch.sum(self.sparsity_mask).item()
        total_params = self.sparsity_mask.numel()
        actual_sparsity = 1.0 - (active_params / total_params)
        
        # Count parameters
        base_params = sum(p.numel() for p in self.base_layer.parameters())
        lora_params = sum(p.numel() for p in [self.lora_A, self.lora_B])
        active_lora_params = lora_params * (1 - actual_sparsity)
        
        return {
            "sparsity": actual_sparsity,
            "active_connections": active_params,
            "total_connections": total_params,
            "base_params": base_params,
            "lora_params": lora_params,
            "active_lora_params": active_lora_params,
            "param_ratio": active_lora_params / base_params
        }

class QwenModelForTraining(torch.nn.Module):
    """
    Wrapper around a PyTorch model to apply SLoRA
    and handle training specifics
    """
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Apply SLoRA to attention layers
        self.apply_slora()
        
        # Print trainable parameters
        self.print_trainable_parameters()
    
    def apply_slora(self):
        """Apply SLoRA to attention layers"""
        # Default target layers are attention query, key, value, output projections
        if not hasattr(self.config, "target_modules"):
            self.config.target_modules = ["query", "key", "value", "output"]
        
        # Apply SLoRA to specific modules
        slora_stats = []
        for name, module in self.base_model.named_modules():
            # Check if this module should be adapted with SLoRA
            for target in self.config.target_modules:
                if target in name:
                    # Find linear layers 
                    for child_name, child in module.named_children():
                        if isinstance(child, torch.nn.Linear):
                            logger.info(f"Applying SLoRA to {name}.{child_name}")
                            
                            # Replace with SLoRA layer
                            slora_layer = SparseLinearLayer(
                                child,
                                rank=self.config.rank,
                                alpha=self.config.alpha,
                                dropout=self.config.dropout,
                                sparsity=self.config.sparsity
                            )
                            
                            # Replace the linear layer with SLoRA layer
                            setattr(module, child_name, slora_layer)
                            
                            # Store stats
                            slora_stats.append(slora_layer.get_sparsity_stats())
        
        # Calculate and log overall stats
        if slora_stats:
            total_base_params = sum(stats["base_params"] for stats in slora_stats)
            total_lora_params = sum(stats["lora_params"] for stats in slora_stats)
            total_active_lora_params = sum(stats["active_lora_params"] for stats in slora_stats)
            
            logger.info(f"Applied SLoRA with sparsity {self.config.sparsity:.2f}")
            logger.info(f"Total base parameters: {total_base_params:,}")
            logger.info(f"Total SLoRA parameters: {total_lora_params:,}")
            logger.info(f"Active SLoRA parameters: {total_active_lora_params:,.2f}")
            logger.info(f"Parameter efficiency ratio: {total_active_lora_params / total_base_params:.6f}")
    
    def print_trainable_parameters(self):
        """Prints the number of trainable parameters"""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        logger.info(
            f"Trainable parameters: {trainable_params:,} ({trainable_params / all_params:.2%})"
        )
    
    def forward(self, **kwargs):
        """Forward pass through the model"""
        return self.base_model(**kwargs)
    
    def save_slora_weights(self, path):
        """Save only the SLoRA weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Extract SLoRA weights
        slora_state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, SparseLinearLayer):
                slora_state_dict[f"{name}.lora_A"] = module.lora_A
                slora_state_dict[f"{name}.lora_B"] = module.lora_B
                slora_state_dict[f"{name}.sparsity_mask"] = module.sparsity_mask
        
        # Save SLoRA weights
        torch.save(slora_state_dict, path)
        logger.info(f"Saved SLoRA weights to {path}")
        
        # Save config
        with open(path.replace(".pt", "_config.json"), "w") as f:
            json.dump({
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "dropout": self.config.dropout,
                "sparsity": self.config.sparsity,
                "target_modules": self.config.target_modules
            }, f, indent=2)

class SimplifiedQwenModel(torch.nn.Module):
    """
    Simplified model for testing and development 
    when actual Qwen model is not available
    """
    def __init__(self, vocab_size=32000, hidden_size=2048, num_layers=24, num_heads=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Embeddings
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            layer = torch.nn.ModuleDict({
                "attention": torch.nn.ModuleDict({
                    "query": torch.nn.Linear(hidden_size, hidden_size),
                    "key": torch.nn.Linear(hidden_size, hidden_size),
                    "value": torch.nn.Linear(hidden_size, hidden_size),
                    "output": torch.nn.Linear(hidden_size, hidden_size)
                }),
                "mlp": torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size * 4),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_size * 4, hidden_size)
                )
            })
            self.layers.append(layer)
        
        # LM Head
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embedding
        hidden_states = self.embeddings(input_ids)
        
        # Process through transformer layers
        for layer in self.layers:
            # Self-attention
            query = layer["attention"]["query"](hidden_states)
            key = layer["attention"]["key"](hidden_states)
            value = layer["attention"]["value"](hidden_states)
            
            # Simple scaled dot-product attention (without real attention mechanism for simplicity)
            attention_output = query + key + value
            attention_output = layer["attention"]["output"](attention_output)
            
            # Add residual connection
            hidden_states = hidden_states + attention_output
            
            # MLP
            mlp_output = layer["mlp"](hidden_states)
            
            # Add residual connection
            hidden_states = hidden_states + mlp_output
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Simple cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return {"logits": logits, "loss": loss}

class SimpleTokenizer:
    """
    Simple tokenizer for development and testing
    In production, you'd use a proper tokenizer with vocabulary
    """
    
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.unk_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 3
        
    def encode(self, text, add_special_tokens=True):
        """Simple encoding - just map characters to token IDs"""
        # This is just a simple implementation - in production use a real tokenizer
        if not text:
            return []
            
        # Very simple char-level encoding
        token_ids = [ord(c) % (self.vocab_size - 4) + 4 for c in text]
        
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
            
        return token_ids
    
    def decode(self, ids):
        """Simple decoding - map token IDs back to characters"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            # Handle nested lists (batch)
            return [self.decode(seq) for seq in ids]
            
        # Remove special tokens
        if ids and ids[0] == self.bos_token_id:
            ids = ids[1:]
        if ids and ids[-1] == self.eos_token_id:
            ids = ids[:-1]
            
        # Convert to string
        text = ""
        for id in ids:
            if id >= 4:  # Skip special tokens
                char_code = (id - 4) % 128  # Keep in ASCII range
                text += chr(char_code)
                
        return text
    
    def __call__(self, texts, padding=True, truncation=True, max_length=1024, return_tensors=None):
        """Tokenize a batch of texts"""
        if isinstance(texts, str):
            texts = [texts]
            
        # Encode all texts
        encoded_texts = [self.encode(text) for text in texts]
        
        # Apply truncation
        if truncation and max_length:
            encoded_texts = [ids[:max_length] for ids in encoded_texts]
            
        # Apply padding
        if padding:
            max_len = max(len(ids) for ids in encoded_texts)
            encoded_texts = [
                ids + [self.pad_token_id] * (max_len - len(ids))
                for ids in encoded_texts
            ]
            
        # Create attention masks
        attention_masks = [
            [1] * len(ids) 
            for ids in encoded_texts
        ]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            input_ids = torch.tensor(encoded_texts)
            attention_mask = torch.tensor(attention_masks)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        elif return_tensors == "np":
            import numpy as np
            return {
                "input_ids": np.array(encoded_texts),
                "attention_mask": np.array(attention_masks)
            }
        else:
            return {
                "input_ids": encoded_texts,
                "attention_mask": attention_masks
            }

class SimpleDataset(Dataset):
    """Simple dataset for testing"""
    def __init__(self, tokenizer, data=None, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Generate simple data if none provided
        if data is None:
            data = []
            for i in range(100):
                data.append(f"Sample text {i} for testing. This is a longer piece of text for training the model.")
        
        self.examples = []
        for text in data:
            tokenized = tokenizer(
                text, 
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            self.examples.append({
                "input_ids": tokenized["input_ids"][0],
                "labels": tokenized["input_ids"][0].clone()
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# Add resource monitoring and optimization functions
def get_memory_usage():
    """Get memory usage statistics for current process"""
    import psutil
    
    # Initialize dictionary
    mem_info = {}
    
    # Get current process
    process = psutil.Process()
    
    # Get RAM usage
    ram_info = psutil.virtual_memory()
    ram_total_gb = ram_info.total / (1024 ** 3)
    ram_used_gb = process.memory_info().rss / (1024 ** 3)
    
    # Store RAM info
    mem_info["ram_total_gb"] = ram_total_gb
    mem_info["ram_used_gb"] = ram_used_gb
    mem_info["cpu_count"] = psutil.cpu_count(logical=True)
    
    # Get GPU info if available
    if torch.cuda.is_available():
        try:
            mem_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
            mem_info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            mem_info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as e:
            print(f"Error getting GPU memory info: {e}")
            mem_info["gpu_allocated_gb"] = 0
            mem_info["gpu_reserved_gb"] = 0
            mem_info["gpu_total_gb"] = 0
            mem_info["gpu_name"] = "Unknown"
    
    return mem_info

def optimize_memory_usage(full_utilization=False):
    """Optimize memory usage for training"""
    # Clear any cached memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if full_utilization:
        # Set PyTorch to use as much GPU memory as possible
        if torch.cuda.is_available():
            # Reserve memory for training
            torch.cuda.empty_cache()
            # Create a dummy tensor to allocate memory then free it
            dummy = torch.ones((1024, 1024, 128), device='cuda')
            del dummy
            torch.cuda.empty_cache()
            
            # Enable cudnn benchmarking for faster training
            torch.backends.cudnn.benchmark = True
        
        # If using CPU, set thread settings for optimal performance
        if not torch.cuda.is_available():
            # Use all available cores, but leave 1 for system processes
            num_cores = max(1, psutil.cpu_count() - 1)
            torch.set_num_threads(num_cores)
            
    else:
        # Use more conservative settings
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
        
    logger.info(f"Memory optimization complete. Current usage:")
    mem_info = get_memory_usage()
    logger.info(f"RAM: {mem_info['ram_used_gb']:.2f}GB / {mem_info['ram_total_gb']:.2f}GB ({mem_info['ram_percent']}%)")
    if torch.cuda.is_available():
        logger.info(f"GPU: {mem_info['gpu_allocated_gb']:.2f}GB allocated, {mem_info['gpu_reserved_gb']:.2f}GB reserved")

# Enhanced SLoRA implementation with adaptive sparsity
class EnhancedSLoRA:
    """Enhanced SLoRA implementation with adaptive sparsity and rank"""
    def __init__(self, model, rank=8, sparsity=0.9, alpha=16, 
                 adaptive_sparsity=False, target_memory_usage=0.9):
        self.model = model
        self.rank = rank
        self.sparsity = sparsity
        self.alpha = alpha
        self.adaptive_sparsity = adaptive_sparsity
        self.target_memory_usage = target_memory_usage
        self.original_weights = {}
        self.lora_weights = {}
        self.lora_masks = {}
        
        # Identify linear layers
        self.linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.linear_layers.append((name, module))
    
    def apply(self):
        """Apply SLoRA to the model"""
        logger.info(f"Applying SLoRA with rank={self.rank}, sparsity={self.sparsity:.2f}")
        
        if self.adaptive_sparsity and torch.cuda.is_available():
            # Calculate adaptive sparsity based on available GPU memory
            self.calculate_adaptive_sparsity()
        
        # Initialize LoRA parameters for each linear layer
        for name, layer in self.linear_layers:
            # Store original weight
            self.original_weights[name] = layer.weight.data.clone()
            
            # Create LoRA weights with small initialization
            in_features, out_features = layer.weight.shape
            lora_a = torch.randn(in_features, self.rank, device=layer.weight.device) * 0.01
            lora_b = torch.zeros(self.rank, out_features, device=layer.weight.device)
            
            self.lora_weights[name] = (lora_a, lora_b)
            
            # Create sparsity mask
            mask = self.create_sparsity_mask(layer.weight.shape)
            self.lora_masks[name] = mask
            
            # Modify layer's forward method to use SLoRA
            self.patch_layer(name, layer)
        
        return self
    
    def calculate_adaptive_sparsity(self):
        """Calculate adaptive sparsity based on available GPU memory"""
        mem_info = get_memory_usage()
        
        if torch.cuda.is_available():
            # Get total parameter count for linear layers
            total_params = 0
            for _, layer in self.linear_layers:
                total_params += layer.weight.numel()
            
            # Calculate memory needed for full LoRA (non-sparse)
            bytes_per_param = 4  # FP32
            if self.model.dtype == torch.float16:
                bytes_per_param = 2  # FP16
            
            full_lora_memory_gb = (total_params * 2 * bytes_per_param) / (1024**3)  # LoRA A & B matrices
            
            # Calculate available memory and target usage
            available_gpu_memory = mem_info["gpu_total_gb"] * self.target_memory_usage - mem_info["gpu_allocated_gb"]
            
            # Calculate required sparsity to fit in memory
            if available_gpu_memory < full_lora_memory_gb:
                required_sparsity = 1.0 - (available_gpu_memory / full_lora_memory_gb)
                # Ensure sparsity is within reasonable bounds
                self.sparsity = min(0.98, max(self.sparsity, required_sparsity))
                logger.info(f"Adaptive sparsity: adjusted to {self.sparsity:.2f} based on available memory")
            else:
                logger.info(f"Adaptive sparsity: keeping original sparsity {self.sparsity:.2f}")
    
    def create_sparsity_mask(self, shape):
        """Create a mask for sparse updates"""
        mask = torch.rand(shape, device=self.model.device) > self.sparsity
        return mask
    
    def patch_layer(self, name, layer):
        """Patch a layer to use SLoRA during forward pass"""
        original_forward = layer.forward
        lora_a, lora_b = self.lora_weights[name]
        mask = self.lora_masks[name]
        
        def lora_forward(x):
            # Compute regular output
            original_output = original_forward(x)
            
            # Add LoRA contribution, apply scaling
            lora_output = torch.mm(x, lora_a).mm(lora_b)
            lora_output = lora_output * (self.alpha / self.rank)
            
            # Apply sparsity mask
            lora_output = lora_output * mask
            
            return original_output + lora_output
        
        # Replace forward method
        layer.forward = lora_forward
    
    def get_trainable_params(self):
        """Get trainable parameters for optimizer"""
        trainable_params = []
        
        for name in self.lora_weights:
            lora_a, lora_b = self.lora_weights[name]
            trainable_params.append(lora_a)
            trainable_params.append(lora_b)
        
        return trainable_params
    
    def save(self, path):
        """Save SLoRA weights"""
        save_dict = {}
        
        for name in self.lora_weights:
            lora_a, lora_b = self.lora_weights[name]
            save_dict[f"{name}.lora_a"] = lora_a
            save_dict[f"{name}.lora_b"] = lora_b
            save_dict[f"{name}.mask"] = self.lora_masks[name]
        
        # Add metadata
        save_dict["_metadata"] = {
            "rank": self.rank,
            "sparsity": self.sparsity,
            "alpha": self.alpha,
        }
        
        # Save to file
        save_file(save_dict, path)
        logger.info(f"SLoRA weights saved to {path}")
    
    def load(self, path):
        """Load SLoRA weights"""
        # Load weights
        loaded_dict = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                loaded_dict[key] = f.get_tensor(key)
        
        # Load metadata
        if "_metadata" in loaded_dict:
            metadata = loaded_dict["_metadata"]
            self.rank = metadata["rank"]
            self.sparsity = metadata["sparsity"]
            self.alpha = metadata["alpha"]
            del loaded_dict["_metadata"]
        
        # Group parameters by layer
        for name, layer in self.linear_layers:
            if f"{name}.lora_a" in loaded_dict and f"{name}.lora_b" in loaded_dict:
                lora_a = loaded_dict[f"{name}.lora_a"]
                lora_b = loaded_dict[f"{name}.lora_b"]
                
                # Store weights
                self.lora_weights[name] = (lora_a, lora_b)
                
                # Load mask if available, otherwise create new one
                if f"{name}.mask" in loaded_dict:
                    self.lora_masks[name] = loaded_dict[f"{name}.mask"]
                else:
                    self.lora_masks[name] = self.create_sparsity_mask(layer.weight.shape)
                
                # Patch layer
                self.patch_layer(name, layer)
        
        logger.info(f"SLoRA weights loaded from {path}")
        return self

# Enhanced training function
def train_model(model, 
                train_dataloader, 
                val_dataloader=None,
                epochs=3, 
                lr=2e-4, 
                fp16=False,
                slora_rank=8,
                slora_sparsity=0.9,
                slora_alpha=16,
                output_dir="./output",
                eval_steps=100,
                save_steps=500,
                gradient_accumulation_steps=1,
                warmup_steps=100,
                adaptive_sparsity=False,
                full_resource_utilization=False):
    """Train the model with SLoRA fine-tuning"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Optimize memory usage
    optimize_memory_usage(full_resource_utilization)
    
    # Apply SLoRA
    slora = EnhancedSLoRA(model, 
                  rank=slora_rank, 
                  sparsity=slora_sparsity, 
                  alpha=slora_alpha,
                  adaptive_sparsity=adaptive_sparsity,
                  target_memory_usage=0.95 if full_resource_utilization else 0.8)
    slora.apply()
    
    # Set up optimizer
    optimizer = AdamW(slora.get_trainable_params(), lr=lr)
    
    # Set up scheduler
    total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
    
    def get_lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
    
    # Set up mixed precision training
    scaler = GradScaler() if fp16 else None
    
    # Training loop
    global_step = 0
    model.train()
    
    # Create arrays to store metrics
    train_losses = []
    eval_losses = []
    learning_rates = []
    
    # Memory usage before training
    mem_before = get_memory_usage()
    
    logger.info(f"Starting training with SLoRA (rank={slora_rank}, sparsity={slora_sparsity:.2f}, alpha={slora_alpha})")
    logger.info(f"Training for {epochs} epochs, {len(train_dataloader)} steps per epoch")
    
    for epoch in range(epochs):
        epoch_iterator = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        tr_loss = 0.0
        
        for step, batch in enumerate(epoch_iterator):
            # Get inputs
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device) if "labels" in batch else input_ids.clone()
            
            # Forward pass with mixed precision if requested
            if fp16:
                with autocast():
                    outputs = model(input_ids)
                    logits = outputs
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        labels.view(-1)
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Update if we've accumulated enough gradients
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            else:
                # Non-mixed precision training
                outputs = model(input_ids)
                logits = outputs
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1)
                )
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update if we've accumulated enough gradients
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            # Track loss and LR
            tr_loss += loss.item() * gradient_accumulation_steps
            
            if global_step % 10 == 0:
                train_losses.append(tr_loss / (step + 1))
                learning_rates.append(scheduler.get_last_lr()[0])
            
            # Update progress bar
            epoch_iterator.set_postfix(loss=tr_loss / (step + 1))
            
            # Evaluation
            if val_dataloader and global_step % eval_steps == 0:
                eval_loss = evaluate_model(model, val_dataloader, fp16=fp16)
                eval_losses.append((global_step, eval_loss))
                logger.info(f"Evaluation at step {global_step}: Loss = {eval_loss:.4f}")
                model.train()  # Set back to training mode
            
            # Save model
            if global_step % save_steps == 0:
                # Save SLoRA weights
                slora_path = os.path.join(output_dir, f"slora_weights_step_{global_step}.safetensors")
                slora.save(slora_path)
                
                # Save training metrics
                save_training_metrics(
                    train_losses, 
                    eval_losses, 
                    learning_rates, 
                    output_dir=output_dir
                )
    
    # Final save
    slora_path = os.path.join(output_dir, "slora_weights_final.safetensors")
    slora.save(slora_path)
    
    # Save training metrics
    save_training_metrics(
        train_losses, 
        eval_losses, 
        learning_rates, 
        output_dir=output_dir
    )
    
    # Calculate memory used during training
    mem_after = get_memory_usage()
    ram_used = mem_after["ram_used_gb"] - mem_before["ram_used_gb"]
    gpu_used = 0
    if torch.cuda.is_available():
        gpu_used = mem_after["gpu_allocated_gb"] - mem_before["gpu_allocated_gb"]
    
    logger.info(f"Training complete. RAM used: {ram_used:.2f} GB, GPU memory used: {gpu_used:.2f} GB")
    
    return {
        "final_loss": tr_loss / len(train_dataloader),
        "global_steps": global_step,
        "slora_config": {
            "rank": slora_rank,
            "sparsity": slora_sparsity,
            "alpha": slora_alpha
        }
    }

# Add a function to save training metrics
def save_training_metrics(train_losses, eval_losses, learning_rates, output_dir):
    """Save and visualize training metrics"""
    # Save raw data
    metrics = {
        "train_losses": train_losses,
        "eval_losses": [{"step": step, "loss": loss} for step, loss in eval_losses],
        "learning_rates": learning_rates
    }
    
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Steps (x10)")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    if eval_losses:
        steps, losses = zip(*eval_losses)
        plt.plot(steps, losses)
        plt.title("Evaluation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()
    
    # Learning rate curve
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Steps (x10)")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "lr_schedule.png"))
    plt.close()

def load_model_from_benchmark(benchmark_settings_path):
    """
    Load the optimal model based on benchmark results
    
    Args:
        benchmark_settings_path: Path to the benchmark settings JSON
        
    Returns:
        QwenModelForTraining model with SLoRA applied
    """
    if not os.path.exists(benchmark_settings_path):
        logger.error(f"Benchmark settings file not found: {benchmark_settings_path}")
        raise FileNotFoundError(f"Benchmark settings file not found: {benchmark_settings_path}")
        
    try:
        # Load benchmark settings
        with open(benchmark_settings_path, 'r') as f:
            settings = json.load(f)
        
        # Extract model info
        model_name = settings.get("model_name", "")
        model_params = settings.get("model_params_billions", 0)
        precision = settings.get("precision", "fp16")
        
        logger.info(f"Loading model: {model_name} ({model_params}B) with precision: {precision}")
        
        # Determine model parameters based on size
        if model_params <= 0.5:
            hidden_size = 1024
            num_layers = 24
            num_heads = 16
        elif model_params <= 1.5:
            hidden_size = 2048
            num_layers = 24
            num_heads = 16
        elif model_params <= 7:
            hidden_size = 4096
            num_layers = 32
            num_heads = 32
        elif model_params <= 14:
            hidden_size = 5120
            num_layers = 40
            num_heads = 40
        else:  # 32B or larger
            hidden_size = 8192
            num_layers = 64
            num_heads = 64
        
        # Create simplified model for testing/development
        # In production, you would download and load the actual model weights
        base_model = SimplifiedQwenModel(
            vocab_size=32000,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # Apply SLoRA
        config = argparse.Namespace(
            rank=settings.get("slora_rank", 16),
            alpha=settings.get("slora_alpha", 32),
            dropout=settings.get("slora_dropout", 0.05),
            sparsity=settings.get("slora_sparsity", 0.95),
            target_modules=["query", "key", "value", "output"]
        )
        
        model = QwenModelForTraining(base_model, config)
        
        # Apply precision
        if precision == "fp16":
            model = model.half()
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        logger.info(f"Successfully loaded model with configuration: {config}")
        
        return model
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing benchmark settings file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from benchmark: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Train Qwen models with PyTorch and SLoRA")
    
    # Existing arguments...
    parser.add_argument("--model_size_billions", type=float, default=None,
                        help="Model size in billions of parameters. Will be overridden by settings file if provided.")
    parser.add_argument("--settings_file", type=str, default=None,
                        help="JSON file with optimal settings from benchmarking")
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "fp16", "int8", "int4"],
                        help="Precision to use. Will be overridden by settings file if provided.")
    parser.add_argument("--context_length", type=int, default=None,
                        help="Context length for training. Will be overridden by settings file if provided.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training. Will be overridden by settings file if provided.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Gradient accumulation steps. Will be overridden by settings file if provided.")
    parser.add_argument("--output_dir", type=str, default="./trained_models",
                        help="Directory to save trained model")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to training data file")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation data file")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Steps between saving checkpoints")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--rank", type=int, default=8,
                        help="SLoRA rank")
    parser.add_argument("--sparsity", type=float, default=0.9,
                        help="SLoRA sparsity (0.0 to 1.0)")
    parser.add_argument("--alpha", type=float, default=16,
                        help="SLoRA scaling factor")
    
    # New arguments
    parser.add_argument("--adaptive_sparsity", action="store_true",
                        help="Adaptively determine sparsity based on available resources")
    parser.add_argument("--full_resource_utilization", action="store_true",
                        help="Maximize resource utilization (may be less stable)")
    
    args = parser.parse_args()
    
    # Load settings from benchmark if available
    settings = {}
    if args.settings_file and os.path.exists(args.settings_file):
        try:
            with open(args.settings_file, "r") as f:
                settings = json.load(f)
                logger.info(f"Loaded settings from {args.settings_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing settings file: {e}")
            return
    
    # Override with explicit arguments if provided
    model_size = args.model_size_billions or settings.get("model_params_billions", 1.5)
    precision = args.precision or settings.get("precision", "fp16")
    context_length = args.context_length or settings.get("context_length", 1024)
    batch_size = args.batch_size or settings.get("batch_size", 1)
    gradient_accumulation_steps = args.gradient_accumulation_steps or settings.get("gradient_accumulation_steps", 16)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log settings
    logger.info("=== Training Settings ===")
    logger.info(f"Model Size: {model_size}B parameters")
    logger.info(f"Precision: {precision}")
    logger.info(f"Context Length: {context_length}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    logger.info(f"SLoRA Rank: {args.rank}")
    logger.info(f"SLoRA Sparsity: {args.sparsity}")
    logger.info(f"SLoRA Alpha: {args.alpha}")
    logger.info(f"Adaptive Sparsity: {args.adaptive_sparsity}")
    logger.info(f"Full Resource Utilization: {args.full_resource_utilization}")
    
    # Load model
    logger.info("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # If we have a settings file, use it to load the model
        if args.settings_file and os.path.exists(args.settings_file):
            model = load_model_from_benchmark(args.settings_file)
        else:
            # Create simplified model with the requested parameters
            logger.info(f"Creating simplified model with {model_size}B parameters")
            # Determine model parameters based on size
            if model_size <= 0.5:
                hidden_size = 1024
                num_layers = 24
                num_heads = 16
            elif model_size <= 1.5:
                hidden_size = 2048
                num_layers = 24
                num_heads = 16
            elif model_size <= 7:
                hidden_size = 4096
                num_layers = 32
                num_heads = 32
            elif model_size <= 14:
                hidden_size = 5120
                num_layers = 40
                num_heads = 40
            else:  # 32B or larger
                hidden_size = 8192
                num_layers = 64
                num_heads = 64
            
            # Create simplified model
            base_model = SimplifiedQwenModel(
                vocab_size=32000,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads
            )
            
            # Apply SLoRA
            config = argparse.Namespace(
                rank=args.rank,
                alpha=args.alpha,
                dropout=0.05,
                sparsity=args.sparsity,
                target_modules=["query", "key", "value", "output"]
            )
            
            model = QwenModelForTraining(base_model, config)
            
            # Apply precision
            if precision == "fp16":
                model = model.half()
            
            # Move to device
            model = model.to(device)
            
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=32000)
    
    # Load or create datasets
    try:
        if args.train_data and os.path.exists(args.train_data):
            # Load real data
            logger.info(f"Loading training data from {args.train_data}")
            train_dataset = SimpleDataset(tokenizer, data=args.train_data, max_length=context_length)
        else:
            # Create dummy dataset for testing
            logger.info("Creating dummy training dataset")
            train_dataset = SimpleDataset(tokenizer, max_length=context_length)
            # Generate some random data
            for i in range(1000):
                train_dataset.data.append(tokenizer.encode(f"Sample text {i}", add_special_tokens=True))
        
        if args.val_data and os.path.exists(args.val_data):
            # Load real validation data
            logger.info(f"Loading validation data from {args.val_data}")
            val_dataset = SimpleDataset(tokenizer, data=args.val_data, max_length=context_length)
        else:
            # Create dummy validation dataset
            logger.info("Creating dummy validation dataset")
            val_dataset = SimpleDataset(tokenizer, max_length=context_length)
            # Generate some random data
            for i in range(100):
                val_dataset.data.append(tokenizer.encode(f"Validation text {i}", add_special_tokens=True))
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        return
    
    # Create dataloaders
    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        return
    
    # Train model
    try:
        use_fp16 = (precision == "fp16")
        training_results = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=args.epochs,
            lr=args.learning_rate,
            fp16=use_fp16,
            slora_rank=args.rank,
            slora_sparsity=args.sparsity,
            slora_alpha=args.alpha,
            output_dir=args.output_dir,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            adaptive_sparsity=args.adaptive_sparsity,
            full_resource_utilization=args.full_resource_utilization
        )
        
        # Save final results
        with open(os.path.join(args.output_dir, "training_results.json"), "w") as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"Training complete. Final loss: {training_results['final_loss']:.4f}")
        logger.info(f"Model saved to {args.output_dir}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return

if __name__ == "__main__":
    main() 