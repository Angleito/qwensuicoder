#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processor for Sui Move code and SDK examples.
"""

import os
import json
import glob
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SuiTrainingDataProcessor:
    def __init__(
        self, 
        raw_data_dir: str, 
        processed_data_dir: str,
        training_data_dir: str,
        validation_data_dir: str,
        validation_split: float = 0.1,
        max_examples: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize the Sui Training Data Processor.
        
        Args:
            raw_data_dir: Directory containing raw data
            processed_data_dir: Directory to save processed data
            training_data_dir: Directory to save training data
            validation_data_dir: Directory to save validation data
            validation_split: Fraction of data to use for validation
            max_examples: Maximum number of examples to process
            random_seed: Random seed for reproducibility
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.training_data_dir = training_data_dir
        self.validation_data_dir = validation_data_dir
        self.validation_split = validation_split
        self.max_examples = max_examples
        self.random_seed = random_seed
        
        random.seed(self.random_seed)
        
        # Ensure directories exist
        for directory in [self.processed_data_dir, self.training_data_dir, self.validation_data_dir]:
            os.makedirs(directory, exist_ok=True)
        
    def process_move_code_examples(self, move_examples_dir: str) -> List[Dict[str, Any]]:
        """
        Process Sui Move code examples into training format.
        
        Args:
            move_examples_dir: Directory containing Move code examples
            
        Returns:
            List of processed examples
        """
        logger.info(f"Processing Move code examples from {move_examples_dir}")
        processed_examples = []
        
        # Find all .move files
        move_files = glob.glob(os.path.join(move_examples_dir, "**", "*.move"), recursive=True)
        logger.info(f"Found {len(move_files)} Move files")
        
        for file_path in tqdm(move_files[:self.max_examples] if self.max_examples else move_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_content = f.read()
                
                # Create a training example
                example = {
                    "text": self._format_move_example(code_content, file_path),
                    "source_file": os.path.basename(file_path),
                    "type": "move_code"
                }
                
                processed_examples.append(example)
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                
        return processed_examples
    
    def process_typescript_examples(self, typescript_examples_dir: str) -> List[Dict[str, Any]]:
        """
        Process Sui TypeScript SDK examples into training format.
        
        Args:
            typescript_examples_dir: Directory containing TypeScript examples
            
        Returns:
            List of processed examples
        """
        logger.info(f"Processing TypeScript examples from {typescript_examples_dir}")
        processed_examples = []
        
        # Find all .ts files
        ts_files = glob.glob(os.path.join(typescript_examples_dir, "**", "*.ts"), recursive=True)
        logger.info(f"Found {len(ts_files)} TypeScript files")
        
        for file_path in tqdm(ts_files[:self.max_examples] if self.max_examples else ts_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_content = f.read()
                
                # Create a training example
                example = {
                    "text": self._format_typescript_example(code_content, file_path),
                    "source_file": os.path.basename(file_path),
                    "type": "typescript_sdk"
                }
                
                processed_examples.append(example)
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                
        return processed_examples
    
    def process_python_examples(self, python_examples_dir: str) -> List[Dict[str, Any]]:
        """
        Process Sui Python SDK examples into training format.
        
        Args:
            python_examples_dir: Directory containing Python examples
            
        Returns:
            List of processed examples
        """
        logger.info(f"Processing Python examples from {python_examples_dir}")
        processed_examples = []
        
        # Find all .py files
        py_files = glob.glob(os.path.join(python_examples_dir, "**", "*.py"), recursive=True)
        logger.info(f"Found {len(py_files)} Python files")
        
        for file_path in tqdm(py_files[:self.max_examples] if self.max_examples else py_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_content = f.read()
                
                # Create a training example
                example = {
                    "text": self._format_python_example(code_content, file_path),
                    "source_file": os.path.basename(file_path),
                    "type": "python_sdk"
                }
                
                processed_examples.append(example)
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                
        return processed_examples
    
    def _format_move_example(self, code: str, file_path: str) -> str:
        """Format Move code example for training"""
        file_name = os.path.basename(file_path)
        return f"### Sui Move Smart Contract\n### Filename: {file_name}\n```move\n{code}\n```"
    
    def _format_typescript_example(self, code: str, file_path: str) -> str:
        """Format TypeScript example for training"""
        file_name = os.path.basename(file_path)
        return f"### Sui TypeScript SDK Example\n### Filename: {file_name}\n```typescript\n{code}\n```"
    
    def _format_python_example(self, code: str, file_path: str) -> str:
        """Format Python example for training"""
        file_name = os.path.basename(file_path)
        return f"### Sui Python SDK Example\n### Filename: {file_name}\n```python\n{code}\n```"
    
    def split_train_validation(self, examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split examples into training and validation sets"""
        random.shuffle(examples)
        split_idx = int(len(examples) * (1 - self.validation_split))
        return examples[:split_idx], examples[split_idx:]
    
    def save_jsonl(self, examples: List[Dict[str, Any]], output_file: str) -> None:
        """Save examples to a JSONL file"""
        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        logger.info(f"Saved {len(examples)} examples to {output_file}")
    
    def process_all_data(self, move_dir: str, typescript_dir: str, python_dir: str) -> None:
        """Process all data and create training/validation splits"""
        # Process all data types
        move_examples = self.process_move_code_examples(move_dir)
        ts_examples = self.process_typescript_examples(typescript_dir)
        py_examples = self.process_python_examples(python_dir)
        
        # Save processed data
        self.save_jsonl(move_examples, os.path.join(self.processed_data_dir, "move_examples.jsonl"))
        self.save_jsonl(ts_examples, os.path.join(self.processed_data_dir, "typescript_examples.jsonl"))
        self.save_jsonl(py_examples, os.path.join(self.processed_data_dir, "python_examples.jsonl"))
        
        # Split and save training/validation data
        move_train, move_val = self.split_train_validation(move_examples)
        ts_train, ts_val = self.split_train_validation(ts_examples)
        py_train, py_val = self.split_train_validation(py_examples)
        
        # Save individual files
        self.save_jsonl(move_train, os.path.join(self.training_data_dir, "move_train.jsonl"))
        self.save_jsonl(move_val, os.path.join(self.validation_data_dir, "move_val.jsonl"))
        self.save_jsonl(ts_train, os.path.join(self.training_data_dir, "typescript_train.jsonl"))
        self.save_jsonl(ts_val, os.path.join(self.validation_data_dir, "typescript_val.jsonl"))
        self.save_jsonl(py_train, os.path.join(self.training_data_dir, "python_train.jsonl"))
        self.save_jsonl(py_val, os.path.join(self.validation_data_dir, "python_val.jsonl"))
        
        # Combine all training data
        all_train = move_train + ts_train + py_train
        all_val = move_val + ts_val + py_val
        random.shuffle(all_train)
        random.shuffle(all_val)
        
        # Save combined files
        self.save_jsonl(all_train, os.path.join(self.training_data_dir, "combined_train.jsonl"))
        self.save_jsonl(all_val, os.path.join(self.validation_data_dir, "combined_val.jsonl"))
        
        logger.info(f"Processed {len(move_examples)} Move examples, "
                  f"{len(ts_examples)} TypeScript examples, and "
                  f"{len(py_examples)} Python examples")
        logger.info(f"Created {len(all_train)} training examples and {len(all_val)} validation examples")

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    parser = argparse.ArgumentParser(description="Process Sui training data")
    parser.add_argument("--move-dir", required=True, help="Directory containing Move code examples")
    parser.add_argument("--typescript-dir", required=True, help="Directory containing TypeScript examples")
    parser.add_argument("--python-dir", required=True, help="Directory containing Python examples")
    parser.add_argument("--raw-data-dir", default="data/raw", help="Directory for raw data")
    parser.add_argument("--processed-data-dir", default="data/processed", help="Directory for processed data")
    parser.add_argument("--training-data-dir", default="data/training", help="Directory for training data")
    parser.add_argument("--validation-data-dir", default="data/validation", help="Directory for validation data")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Fraction of data for validation")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum examples to process per type")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    processor = SuiTrainingDataProcessor(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        training_data_dir=args.training_data_dir,
        validation_data_dir=args.validation_data_dir,
        validation_split=args.validation_split,
        max_examples=args.max_examples,
        random_seed=args.random_seed
    )
    
    processor.process_all_data(args.move_dir, args.typescript_dir, args.python_dir) 