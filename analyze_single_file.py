#!/usr/bin/env python3
"""Analyze a single code file using Ollama"""

import sys
import os
import subprocess
import argparse
import time

def analyze_file(file_path, model="qwen-coder-1.5b", output=None, timeout=90):
    """Analyze a single file using Ollama"""
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            print(f"Warning: File is empty: {file_path}")
            return False
        
        # Build a prompt for code analysis
        prompt = f"""
        Analyze this code for syntax errors, bugs, and code quality issues:
        
        ```
        {content}
        ```
        
        File: {file_path}
        
        Please provide:
        1. A brief description of what this code does
        2. Any potential issues or bugs
        3. Suggestions for improvements
        """
        
        # Run Ollama
        print(f"Analyzing {file_path} with {model}...")
        
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                print(f"Error running Ollama: {result.stderr}")
                return False
            
            analysis = result.stdout.strip()
            
            # Save to output file if specified
            if output:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(f"# Code Analysis: {os.path.basename(file_path)}\n\n")
                    f.write(f"**File:** {file_path}\n\n")
                    f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("## Analysis\n\n")
                    f.write(analysis)
                print(f"Analysis saved to {output}")
            else:
                print("\n--- ANALYSIS ---\n")
                print(analysis)
                print("\n---------------\n")
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"Error: Analysis timed out after {timeout} seconds")
            return False
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Analyze a single code file using Ollama")
    parser.add_argument("file", help="File to analyze")
    parser.add_argument("--model", "-m", help="Ollama model to use", default="qwen-coder-1.5b")
    parser.add_argument("--output", "-o", help="Output file (markdown format)")
    parser.add_argument("--timeout", "-t", help="Timeout in seconds", type=int, default=90)
    args = parser.parse_args()
    
    success = analyze_file(args.file, args.model, args.output, args.timeout)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 