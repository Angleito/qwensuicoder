#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for the Qwen-Sui-Coder model.
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Evaluation metrics
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class QwenSuiEvaluator:
    """
    Evaluator for Qwen-Sui-Coder model on Sui Move tasks.
    """
    
    def __init__(
        self, 
        model_path: str, 
        test_file: str,
        output_file: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the fine-tuned model
            test_file: Path to test data file (JSONL format)
            output_file: Path to save evaluation results
            max_length: Maximum token length for generation
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
        """
        self.model_path = model_path
        self.test_file = test_file
        self.output_file = output_file
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )
        
        # Initialize ROUGE and BLEU metrics
        self.rouge = Rouge()
        
        # Download NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test data from JSONL file.
        
        Returns:
            List of test examples
        """
        examples = []
        with open(self.test_file, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line.strip())
                examples.append(example)
        
        logger.info(f"Loaded {len(examples)} test examples from {self.test_file}")
        return examples
    
    def prepare_prompt(self, prompt_text: str) -> str:
        """
        Format the prompt for model input.
        
        Args:
            prompt_text: Raw prompt text
            
        Returns:
            Formatted prompt
        """
        # Add system prompt for Sui Move code completion
        system_prompt = "You are an AI assistant specialized in Sui Move smart contract development. Write high-quality, secure, and efficient Sui Move code."
        
        # Combine with user prompt
        combined_prompt = f"{system_prompt}\n\n{prompt_text}"
        return combined_prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract only the generated part (not the prompt)
        full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = full_response[len(prompt):]
        
        return response.strip()
    
    def compute_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute evaluation metrics between reference and generated text.
        
        Args:
            reference: Reference code
            hypothesis: Generated code
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # ROUGE scores
        try:
            rouge_scores = self.rouge.get_scores(hypothesis, reference)[0]
            metrics["rouge1_f"] = rouge_scores["rouge-1"]["f"]
            metrics["rouge2_f"] = rouge_scores["rouge-2"]["f"]
            metrics["rougeL_f"] = rouge_scores["rouge-l"]["f"]
        except Exception as e:
            logger.warning(f"Error computing ROUGE: {e}")
            metrics["rouge1_f"] = 0.0
            metrics["rouge2_f"] = 0.0
            metrics["rougeL_f"] = 0.0
        
        # BLEU score
        try:
            ref_tokens = nltk.word_tokenize(reference)
            hyp_tokens = nltk.word_tokenize(hypothesis)
            
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
            metrics["bleu"] = bleu_score
        except Exception as e:
            logger.warning(f"Error computing BLEU: {e}")
            metrics["bleu"] = 0.0
        
        return metrics
    
    def extract_code_from_completion(self, completion: str) -> str:
        """
        Extract code from completion, handling markdown code blocks.
        
        Args:
            completion: Generated completion
            
        Returns:
            Extracted code
        """
        # Check if the completion contains markdown code blocks
        if "```" in completion:
            # Extract code between markdown code blocks
            parts = completion.split("```")
            if len(parts) >= 3:  # At least one code block
                # Take the content of the first code block
                code_block = parts[1]
                
                # If the code block starts with a language identifier (e.g., 'move'), remove it
                lines = code_block.strip().split("\n")
                if lines[0].lower() in ["move", "typescript", "python", "ts", "py"]:
                    code = "\n".join(lines[1:])
                else:
                    code = code_block.strip()
                
                return code
        
        # If no code blocks found, return the whole completion
        return completion.strip()
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on test examples.
        
        Returns:
            Evaluation results
        """
        examples = self.load_test_data()
        results = []
        
        # Track metrics
        all_metrics = {
            "rouge1_f": [],
            "rouge2_f": [],
            "rougeL_f": [],
            "bleu": []
        }
        
        for idx, example in enumerate(tqdm(examples, desc="Evaluating")):
            # Extract prompt from the example
            # This may need to be adjusted based on your test data format
            if "prompt" in example:
                prompt_text = example["prompt"]
            elif "instruction" in example:
                prompt_text = example["instruction"]
            else:
                # Extract the first part of the text before code block as prompt
                text = example["text"]
                if "```" in text:
                    prompt_text = text.split("```")[0].strip()
                else:
                    prompt_text = "Complete the following Sui Move code:"
            
            # Format prompt
            formatted_prompt = self.prepare_prompt(prompt_text)
            
            # Generate completion
            completion = self.generate_response(formatted_prompt)
            
            # Extract code from completion
            generated_code = self.extract_code_from_completion(completion)
            
            # Extract reference code from example
            if "reference" in example:
                reference_code = example["reference"]
            else:
                # Extract code from the example text
                text = example["text"]
                if "```" in text:
                    parts = text.split("```")
                    if len(parts) >= 3:
                        reference_code = parts[1]
                        lines = reference_code.strip().split("\n")
                        if lines[0].lower() in ["move", "typescript", "python", "ts", "py"]:
                            reference_code = "\n".join(lines[1:])
                else:
                    # No code block found, use empty string
                    reference_code = ""
            
            # Compute metrics
            metrics = self.compute_metrics(reference_code, generated_code)
            
            # Add result
            result = {
                "id": idx,
                "prompt": prompt_text,
                "completion": completion,
                "generated_code": generated_code,
                "reference_code": reference_code,
                "metrics": metrics
            }
            results.append(result)
            
            # Update metric trackers
            for metric, value in metrics.items():
                all_metrics[metric].append(value)
        
        # Compute average metrics
        avg_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
        
        # Save results
        output = {
            "model": self.model_path,
            "examples": results,
            "metrics": avg_metrics
        }
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {self.output_file}")
        logger.info(f"Average metrics: {avg_metrics}")
        
        return output

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen-Sui-Coder model")
    parser.add_argument("--model-path", required=True, help="Path to the fine-tuned model")
    parser.add_argument("--test-file", required=True, help="Path to test data file (JSONL format)")
    parser.add_argument("--output-file", required=True, help="Path to save evaluation results")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum token length for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    evaluator = QwenSuiEvaluator(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output_file,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    evaluator.evaluate()

if __name__ == "__main__":
    main() 