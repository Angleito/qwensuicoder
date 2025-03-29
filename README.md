# QwenSuiCoder

A project to fine-tune Qwen 2.5 14B Coder model for Sui Move smart contract development and SDK usage.

## Project Overview

QwenSuiCoder aims to create a specialized LLM capable of:
- Writing and debugging Sui Move smart contracts
- Working with Sui SDKs in TypeScript and Python
- Following best practices for the Sui blockchain ecosystem

## Project Structure

```
qwensuicoder/
├── benchmark_qwen_coder.py   # Benchmark script for Qwen 2.5 Coder 14B
├── train_qwen_coder.py       # Optimized training script for Qwen 2.5 Coder 14B
├── run_qwen_training.sh      # All-in-one script for benchmarking and training
├── config/                   # Configuration files for model training
├── data/
│   ├── raw/                  # Raw training data
│   ├── processed/            # Processed data ready for training
│   ├── training/             # Training datasets
│   └── validation/           # Validation datasets
├── examples/
│   ├── move/                 # Example Sui Move smart contracts
│   ├── typescript/           # TypeScript SDK usage examples
│   └── python/               # Python SDK usage examples
├── scripts/                  # Utility scripts for data preparation and training
└── src/
    ├── training/             # Training code
    ├── evaluation/           # Evaluation metrics and testing
    └── utils/                # Utility functions
```

## Setup

### Prerequisites

- Python 3.9+
- PyTorch
- Transformers
- DeepSpeed
- PEFT (Parameter-Efficient Fine-Tuning)
- Hugging Face account with access to Qwen models

### Environment Setup

```bash
# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Setup TypeScript environment (if working with TS examples)
npm install
```

### Hugging Face Access

To access the Qwen models for training, you'll need to:

1. Create a [Hugging Face account](https://huggingface.co/join)
2. Accept the terms of use for the [Qwen2.5 Coder model](https://huggingface.co/Qwen/Qwen2.5-14B-Coder)
3. Generate an [access token](https://huggingface.co/settings/tokens)
4. Set the token in your environment:
   ```bash
   export HF_TOKEN=your_token_here
   ```
   
   Alternatively, create a token file:
   ```bash
   mkdir -p ~/.huggingface
   echo "your_token_here" > ~/.huggingface/token
   ```

## Benchmarking and Training

We've implemented a streamlined workflow to benchmark your hardware and train the model with optimal settings:

### All-in-One Script

The easiest way to get started is with our all-in-one script:

```bash
./run_qwen_training.sh
```

This script will:
1. Check prerequisites
2. Run the benchmark to determine optimal settings for your hardware
3. Train the model using the benchmark results and DeepSpeed optimizations

### Manual Process

If you prefer to run the steps manually:

1. **Run the benchmark to find optimal settings**:
   ```bash
   python benchmark_qwen_coder.py --output_dir benchmark_results
   ```
   This tests Qwen 2.5 Coder 14B with different quantization methods and context lengths.

2. **Train with optimized settings**:
   ```bash
   python train_qwen_coder.py \
     --model_name "Qwen/Qwen2.5-14B-Coder" \
     --auto_optimize \
     --settings_file "benchmark_results/optimal_settings.json" \
     --use_lora \
     --deepspeed
   ```

## Using the Trained Model

After training is complete, you can use the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./trained_models/qwen_coder_14b")
model = AutoModelForCausalLM.from_pretrained("./trained_models/qwen_coder_14b")

# Generate Sui Move code
prompt = "Write a Sui Move smart contract for an NFT collection"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=2048)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## License

MIT 