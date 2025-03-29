#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility to clear CUDA memory and optimize for large model loading
Supports both standard and aggressive cache clearing
"""

import gc
import argparse
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_cuda_cache(aggressive=False):
    """
    Clear CUDA cache to free up memory
    
    Args:
        aggressive: Whether to use aggressive clearing techniques
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.info("CUDA not available, no cache to clear")
            return False
        
        # Get initial memory stats
        init_allocated = torch.cuda.memory_allocated() / (1024**3)
        init_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        logger.info(f"Initial CUDA memory: {init_allocated:.2f}GB allocated, {init_reserved:.2f}GB reserved")
        
        # Run garbage collection first
        gc.collect()
        
        # Basic cache clearing
        torch.cuda.empty_cache()
        
        # More aggressive clearing if requested
        if aggressive:
            logger.info("Performing aggressive memory cleanup...")
            
            # Create and delete a dummy tensor to trigger memory cleanup
            dummy = torch.ones(1, device='cuda')
            del dummy
            
            # Force synchronization
            torch.cuda.synchronize()
            
            # Wait a moment to let cleanup complete
            time.sleep(1)
            
            # Force another garbage collection pass
            gc.collect()
            torch.cuda.empty_cache()
            
            # Try to reduce fragmentation by allocating and freeing a large tensor
            try:
                # Get current free memory
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                
                # Allocate 90% of free memory
                allocation_size = int(0.9 * free_memory)
                if allocation_size > 0:
                    logger.info(f"Allocating {allocation_size / (1024**3):.2f}GB to reduce fragmentation")
                    dummy = torch.empty(allocation_size, dtype=torch.uint8, device='cuda')
                    del dummy
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Fragmentation reduction failed: {e}")
        
        # Get final memory stats
        final_allocated = torch.cuda.memory_allocated() / (1024**3)
        final_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        logger.info(f"Final CUDA memory: {final_allocated:.2f}GB allocated, {final_reserved:.2f}GB reserved")
        logger.info(f"Freed: {init_allocated - final_allocated:.2f}GB allocated, {init_reserved - final_reserved:.2f}GB reserved")
        
        return True
        
    except ImportError:
        logger.warning("PyTorch not available, unable to clear CUDA cache")
        return False
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return False

def optimize_linux_gpu_memory():
    """Try to optimize GPU memory on Linux systems"""
    try:
        # Only applicable on Linux
        if os.name != 'posix':
            return False
        
        # Check for nvidia-smi
        if os.system('which nvidia-smi > /dev/null 2>&1') != 0:
            return False
        
        logger.info("Optimizing NVIDIA settings on Linux...")
        
        # Set compute mode to exclusive process (might require sudo)
        try:
            os.system('nvidia-smi -c 3 > /dev/null 2>&1')
            logger.info("Set compute mode to exclusive process")
        except:
            pass
            
        # Try to disable ECC if available (might require sudo)
        try:
            os.system('nvidia-smi --ecc-config=0 > /dev/null 2>&1')
            logger.info("Disabled ECC memory if supported")
        except:
            pass
            
        return True
    except Exception as e:
        logger.warning(f"Error optimizing Linux GPU settings: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Clear CUDA cache")
    parser.add_argument("--aggressive", action="store_true", help="Use aggressive memory cleanup")
    parser.add_argument("--optimize-linux", action="store_true", help="Try to optimize Linux GPU settings")
    
    args = parser.parse_args()
    
    logger.info("Starting CUDA memory cleanup...")
    
    if args.optimize_linux:
        optimize_linux_gpu_memory()
    
    success = clear_cuda_cache(aggressive=args.aggressive)
    
    if success:
        logger.info("CUDA cache cleared successfully")
    else:
        logger.warning("Failed to clear CUDA cache or CUDA not available")

if __name__ == "__main__":
    main() 