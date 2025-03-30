#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p docs

# Define the output file
OUTPUT_FILE="docs/bluefin_trading_api_guide.md"

# Create a comprehensive prompt for the model
PROMPT="Create a comprehensive guide for automated trading using Bluefin TS API. Include the following sections:

1. Introduction to Bluefin API for algorithmic trading
2. Setup and authentication
3. Basic trading operations (with code examples):
   - Initializing the client
   - Getting account information and balances
   - Creating market and limit orders
   - Canceling orders
   - Reading positions
4. Implementing trading strategies:
   - How to set up a simple trailing stop-loss strategy
   - Implementing a basic grid trading bot
   - Executing flash loans for arbitrage
5. WebSocket connections for real-time data:
   - Setting up price feeds
   - Order book monitoring
   - Position updates
6. Risk management techniques:
   - Position sizing
   - Stop-loss implementation
   - Exposure limits

For each section, provide TypeScript code examples that are production-ready, properly typed, and follow best practices."

echo "Generating Bluefin trading API guide..."
echo "Using model: qwen-coder-1.5b"
echo "Output will be saved to: $OUTPUT_FILE"

# Run the model with the prompt and save to file
{
  echo "# Bluefin Trading API Guide for Automated Trading"
  echo ""
  echo "Generated on $(date)"
  echo ""
  echo "---"
  echo ""
  
  # Run ollama with our prompt
  ollama run qwen-coder-1.5b "$PROMPT"
} > "$OUTPUT_FILE"

echo "✅ Guide generated successfully at $OUTPUT_FILE" 