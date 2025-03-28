# QwenSuiCoder

A project to fine-tune Qwen 2.5 14B model for Sui Move smart contract development and SDK usage.

## Project Overview

QwenSuiCoder aims to create a specialized LLM capable of:
- Writing and debugging Sui Move smart contracts
- Working with Sui SDKs in TypeScript and Python
- Following best practices for the Sui blockchain ecosystem

## Project Structure

```
qwensuicoder/
├── config/                 # Configuration files for model training
├── data/
│   ├── raw/                # Raw training data
│   ├── processed/          # Processed data ready for training
│   ├── training/           # Training datasets
│   └── validation/         # Validation datasets
├── examples/
│   ├── move/               # Example Sui Move smart contracts
│   ├── typescript/         # TypeScript SDK usage examples
│   └── python/             # Python SDK usage examples
├── scripts/                # Utility scripts for data preparation and training
└── src/
    ├── training/           # Training code
    ├── evaluation/         # Evaluation metrics and testing
    └── utils/              # Utility functions
```

## Setup

1. **Prerequisites**
   - Ubuntu on WSL
   - Ollama with Qwen 2.5 14B model
   - Python 3.9+
   - Node.js and npm
   - Sui Move CLI and dependencies

2. **Environment Setup**
   ```bash
   # Create and activate a Python virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Setup TypeScript environment
   npm install
   ```

## Training Process

1. Prepare training data from Sui Move examples and SDK usage
2. Process and format data for fine-tuning
3. Configure training parameters
4. Run fine-tuning on Qwen 2.5 14B
5. Evaluate model performance
6. Iterate and improve

## Usage

Documentation for using the trained model will be added upon completion of initial training.

## License

MIT 