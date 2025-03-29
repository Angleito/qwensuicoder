#!/bin/bash

# Simple script to demonstrate the model
echo "Running Qwen2.5-1.5B-Instruct model demo (smaller parameter model)"
echo "=================================================================="

# Create a directory for output
mkdir -p demo_output

# Run with different programming prompts
echo "Running demo with a palindrome function prompt..."
./simple_model_demo.py --prompt "Write a function to check if a string is a palindrome in Python." --max_tokens 500 | tee demo_output/palindrome_function.txt

echo ""
echo "Running demo with a binary search prompt..."
./simple_model_demo.py --prompt "Implement a binary search algorithm in Python" --max_tokens 500 | tee demo_output/binary_search.txt

echo ""
echo "Running demo with a custom prompt..."
./simple_model_demo.py --prompt "Create a simple Flask API that returns JSON data" --max_tokens 500 | tee demo_output/flask_api.txt

echo ""
echo "Demo complete! All outputs saved to demo_output/ directory." 