# QwenSuiCoder Setup Guide

This document provides instructions on how to set up and use the QwenSuiCoder project, which fine-tunes the Qwen 2.5 14B model to work with Sui Move smart contracts and SDKs.

## Prerequisites

1. **Ubuntu on WSL** (Windows Subsystem for Linux)
2. **Ollama** with Qwen 2.5 14B model installed
3. **Python 3.9+**
4. **Node.js and npm**
5. **Sui Move CLI** (for creating and testing Move code examples)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/qwensuicoder.git
   cd qwensuicoder
   ```

2. **Create a Python virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

5. **Make scripts executable**:
   ```bash
   chmod +x scripts/check_ollama.py scripts/train_pipeline.sh
   ```

## Project Structure

```
qwensuicoder/
├── config/                 # Configuration files
│   └── training_config.yaml    # Model training configuration
├── data/                   # Data directories
│   ├── raw/                # Raw training data
│   ├── processed/          # Processed data
│   ├── training/           # Training datasets
│   └── validation/         # Validation datasets
├── examples/               # Example code
│   ├── move/               # Sui Move examples
│   ├── typescript/         # TypeScript SDK examples
│   └── python/             # Python SDK examples
├── scripts/                # Utility scripts
│   ├── check_ollama.py     # Script to check Ollama availability
│   └── train_pipeline.sh   # Full training pipeline script
└── src/                    # Source code
    ├── training/           # Training code
    │   └── train.py        # Main training script
    ├── evaluation/         # Evaluation code
    │   └── evaluate.py     # Evaluation script
    └── utils/              # Utilities
        └── data_processor.py # Data processing script
```

## Usage

### 1. Check Ollama Setup

First, check if Ollama is properly set up with Qwen:

```bash
python scripts/check_ollama.py
```

If Qwen is not available, you need to pull it with Ollama:

```bash
ollama pull qwen:2.5-14b
```

### 2. Prepare Training Data

You need to collect Sui Move code examples and SDK usage examples:

1. Place Move smart contract examples in `examples/move/`
2. Place TypeScript SDK examples in `examples/typescript/`
3. Place Python SDK examples in `examples/python/`

You can use the provided examples as a starting point, but you'll want more data for real training.

### 3. Run the Training Pipeline

To run the complete training pipeline:

```bash
./scripts/train_pipeline.sh
```

This will:
1. Check Ollama and Qwen availability
2. Process the training data
3. Train the model using LoRA fine-tuning
4. Evaluate the model on validation data

### 4. Manual Steps

If you prefer to run steps individually:

1. **Process data**:
   ```bash
   python src/utils/data_processor.py \
     --move-dir examples/move \
     --typescript-dir examples/typescript \
     --python-dir examples/python
   ```

2. **Train model**:
   ```bash
   python src/training/train.py config/training_config.yaml
   ```

3. **Evaluate model**:
   ```bash
   python src/evaluation/evaluate.py \
     --model-path trained_models \
     --test-file data/validation/combined_val.jsonl \
     --output-file trained_models/evaluation_results.json
   ```

## Collecting Training Data

For effective training, you'll need substantial amounts of Sui Move code and SDK examples:

1. **Sui Move Smart Contracts**:
   - Official Sui example repositories
   - Open-source Sui Move projects
   - Sui documentation examples

2. **TypeScript SDK Usage**:
   - Sui TypeScript SDK documentation
   - Open-source Sui dApps using TypeScript
   - SDK usage patterns and best practices

3. **Python SDK Usage**:
   - Sui Python SDK examples
   - Python integration examples with Sui

## Customizing Training

To customize the training process:

1. Modify `config/training_config.yaml` to adjust training parameters
2. Edit data processing in `src/utils/data_processor.py` if needed
3. Adjust model architecture parameters in `src/training/train.py`

## Troubleshooting

- **Ollama Issues**: Ensure Ollama server is running with `ollama serve`
- **CUDA/GPU Issues**: Adjust the `device_map` parameter in training scripts
- **Memory Issues**: Reduce batch size or use gradient accumulation
- **Training Data Format**: Ensure your examples follow the format in the example files

## Next Steps

After training your model:

1. Test it on new Sui Move code tasks
2. Integrate it with your development environment
3. Continue to collect more training data for better results
4. Consider fine-tuning on specific Sui Move patterns or applications 