#!/usr/bin/env python3
"""Code analysis script using Ollama with Qwen model"""

import sys
import os
import subprocess
import json
import argparse
from pathlib import Path
import time
import textwrap

def get_file_list(directory, exclude_dirs=None, include_extensions=None):
    """Get list of files to analyze, with filtering options."""
    if exclude_dirs is None:
        exclude_dirs = ['node_modules', '.git', 'venv', '__pycache__', 'build', 'dist']
    
    if include_extensions is None:
        include_extensions = ['.py', '.js', '.ts', '.sol', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp']
    
    files = []
    
    for root, dirs, filenames in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if any(file_path.endswith(ext) for ext in include_extensions):
                files.append(file_path)
    
    return files

def chunk_file_content(content, max_chunk_size=2000, overlap=100):
    """Split file content into chunks that fit within model context window."""
    lines = content.splitlines()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, line in enumerate(lines):
        line_size = len(line)
        
        if current_size + line_size > max_chunk_size and current_chunk:
            chunks.append((i - len(current_chunk), i - 1, '\n'.join(current_chunk)))
            # Keep some overlap for context
            overlap_lines = min(len(current_chunk), overlap // (line_size or 1))
            current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
            current_size = sum(len(l) for l in current_chunk)
        
        current_chunk.append(line)
        current_size += line_size
    
    if current_chunk:
        chunks.append((len(lines) - len(current_chunk), len(lines) - 1, '\n'.join(current_chunk)))
    
    return chunks

def analyze_file(file_path, model="qwen-coder-1.5b", timeout=90):
    """Analyze a single file using Ollama with chunking for large files."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            return f"## {file_path}\n\nEmpty file, skipping analysis.\n\n"
        
        # For large files, split into chunks
        if len(content) > 4000:
            chunks = chunk_file_content(content)
            all_analyses = []
            
            for start_line, end_line, chunk_content in chunks:
                chunk_analysis = analyze_chunk(file_path, chunk_content, start_line, end_line, model, timeout)
                all_analyses.append(chunk_analysis)
            
            # Combine all chunk analyses
            combined_analysis = f"## {file_path}\n\n"
            issues_found = False
            
            for analysis in all_analyses:
                if "No issues found" not in analysis:
                    issues_found = True
                    # Extract just the issues section
                    if "### Issues Found" in analysis:
                        issues_section = analysis.split("### Issues Found")[1].strip()
                        combined_analysis += f"### Issues Found (Lines {start_line+1}-{end_line+1})\n\n{issues_section}\n\n"
            
            if not issues_found:
                combined_analysis += "No issues found in this file.\n\n"
            
            return combined_analysis
        else:
            return analyze_chunk(file_path, content, 0, content.count('\n'), model, timeout)
    
    except Exception as e:
        return f"## {file_path}\n\nError analyzing file: {str(e)}\n\n"

def analyze_chunk(file_path, content, start_line, end_line, model="qwen-coder-1.5b", timeout=90):
    """Analyze a chunk of a file using Ollama."""
    try:
        if not content.strip():
            return "No issues found in this section."
        
        # Build a more focused prompt for code analysis
        prompt = textwrap.dedent(f"""
        Analyze this code for syntax errors, bugs, and code quality issues:
        
        ```
        {content}
        ```
        
        File: {file_path}
        Lines: {start_line+1} to {end_line+1}
        
        Provide only the most important issues in this format:
        
        ### Issues Found
        
        1. [Line X] - Description of issue
           ```
           problematic code snippet
           ```
           **Fix**: How to fix the issue
        
        If no issues are found, respond with exactly "No issues found in this section."
        """).strip()
        
        # Run Ollama with increased timeout
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                return f"Error analyzing chunk: {result.stderr}"
            
            # Clean up the output
            analysis = result.stdout.strip()
            # If section doesn't have the heading, add it
            if analysis and "No issues found" not in analysis and "### Issues Found" not in analysis:
                analysis = "### Issues Found\n\n" + analysis
                
            return analysis
            
        except subprocess.TimeoutExpired:
            return f"Timeout analyzing chunk (lines {start_line+1}-{end_line+1})"
        
    except Exception as e:
        return f"Error analyzing chunk: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Analyze code files for issues using Ollama")
    parser.add_argument("directory", help="Directory containing code to analyze")
    parser.add_argument("--output", "-o", help="Output markdown file", default="code_analysis.md")
    parser.add_argument("--model", "-m", help="Ollama model to use", default="qwen-coder-1.5b")
    parser.add_argument("--exclude", "-e", help="Comma-separated list of directories to exclude")
    parser.add_argument("--include", "-i", help="Comma-separated list of file extensions to include")
    parser.add_argument("--timeout", "-t", help="Timeout in seconds for each analysis", type=int, default=90)
    args = parser.parse_args()
    
    # Process arguments
    directory = os.path.expanduser(args.directory)
    output_file = os.path.expanduser(args.output)
    model = args.model
    timeout = args.timeout
    
    exclude_dirs = args.exclude.split(',') if args.exclude else None
    include_extensions = args.include.split(',') if args.include else None
    
    # Check if Ollama is installed and running
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
    except Exception as e:
        print(f"Error: Ollama not installed or not running. {str(e)}")
        sys.exit(1)
    
    # Get files to analyze
    print(f"Scanning directory: {directory}")
    files = get_file_list(directory, exclude_dirs, include_extensions)
    
    if not files:
        print("No files found to analyze.")
        sys.exit(0)
    
    print(f"Found {len(files)} files to analyze")
    
    # Create or clear output file
    with open(output_file, 'w') as f:
        f.write(f"# Code Analysis Report\n\n")
        f.write(f"Directory: {directory}\n\n")
        f.write(f"Analysis performed on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"Total files analyzed: {len(files)}\n\n")
        f.write(f"## Detailed Analysis\n\n")
    
    # Analyze files one by one
    total_files = len(files)
    issues_found = 0
    
    for i, file_path in enumerate(files, 1):
        rel_path = os.path.relpath(file_path, directory)
        print(f"Analyzing {i}/{total_files}: {rel_path}")
        
        analysis = analyze_file(file_path, model, timeout)
        
        # Count issues (simple heuristic)
        if "No issues found" not in analysis:
            issues_found += 1
        
        # Append to output file
        with open(output_file, 'a') as f:
            f.write(analysis)
    
    # Update summary
    with open(output_file, 'r') as f:
        content = f.read()
    
    content = content.replace("Total files analyzed: {len(files)}", 
                             f"Total files analyzed: {total_files}\nFiles with issues: {issues_found}")
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\nAnalysis complete. Results saved to {output_file}")
    print(f"Files analyzed: {total_files}")
    print(f"Files with issues: {issues_found}")

if __name__ == "__main__":
    main()