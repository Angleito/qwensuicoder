#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to verify Ollama with Qwen is working correctly and generate a test response.
"""

import sys
import json
import argparse
import requests
from typing import Dict, Any, Optional

def check_ollama_status() -> bool:
    """
    Check if Ollama server is running.
    
    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("✅ Ollama server is running")
            return True
        else:
            print(f"❌ Ollama server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama server is not running")
        return False

def list_available_models() -> Dict[str, Any]:
    """
    List all available models in Ollama.
    
    Returns:
        Dictionary of available models
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("\nAvailable models:")
            for model in models.get("models", []):
                name = model.get("name", "Unknown")
                size = model.get("size", 0) / (1024 * 1024 * 1024)  # Convert to GB
                print(f"  - {name} ({size:.2f} GB)")
            return models
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            return {}
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama server")
        return {}

def check_qwen_availability() -> bool:
    """
    Check if Qwen model is available in Ollama.
    
    Returns:
        True if Qwen is available, False otherwise
    """
    models = list_available_models()
    model_names = [model.get("name", "") for model in models.get("models", [])]
    
    # Look for Qwen 2.5 model
    qwen_models = [name for name in model_names if "qwen" in name.lower()]
    
    if qwen_models:
        print("\n✅ Found Qwen models:")
        for model in qwen_models:
            print(f"  - {model}")
        return True
    else:
        print("\n❌ No Qwen models found")
        print("To pull the Qwen model, run: ollama pull qwen:2.5-14b")
        return False

def generate_test_response(model_name: str, prompt: str) -> Optional[str]:
    """
    Generate a test response using the specified model.
    
    Args:
        model_name: Name of the model to use
        prompt: Prompt to generate response for
        
    Returns:
        Generated response or None if generation failed
    """
    print(f"\nGenerating test response using {model_name}...")
    print(f"Prompt: {prompt}")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False}
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            print("\nResponse:")
            print(response_text)
            return response_text
        else:
            print(f"❌ Failed to generate response: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama server")
        return None

def main():
    parser = argparse.ArgumentParser(description="Check if Ollama with Qwen is working correctly")
    parser.add_argument("--model", default="qwen2.5:14b", help="Model name to test (default: qwen2.5:14b)")
    parser.add_argument("--prompt", default="Write a simple Sui Move smart contract for a counter.", 
                       help="Test prompt to generate a response")
    
    args = parser.parse_args()
    
    # Check if Ollama is running
    if not check_ollama_status():
        print("\nPlease start Ollama server with: ollama serve")
        sys.exit(1)
    
    # Check if Qwen is available
    check_qwen_availability()
    
    # Generate test response
    response = generate_test_response(args.model, args.prompt)
    
    if response:
        print("\n✅ Successfully generated a response using the model")
        
        # Check if response contains Sui Move code
        if "module" in response.lower() and "struct" in response.lower():
            print("✅ Response appears to contain Sui Move code")
        else:
            print("⚠️ Response may not contain Sui Move code")
    else:
        print("\n❌ Failed to generate a response")

if __name__ == "__main__":
    main() 