# QwenSuiCoder Model Demo

This is a simple demonstration of running language models for code generation on CPU. We've created a simplified version that runs efficiently without requiring a GPU.

## Files

- `simple_model_demo.py`: Python script to load a small language model and generate code from prompts
- `run_model_demo.sh`: Shell script to run the demo with different prompts
- `demo_output/`: Directory containing the generated outputs

## How to Run

```bash
# Run the demo script with multiple prompts
./run_model_demo.sh

# Or run individual prompts
./simple_model_demo.py --prompt "Your programming prompt here" --max_tokens 500
```

## Model Used

The demo uses Qwen/Qwen2.5-1.5B-Instruct, a smaller version of the Qwen 2.5 family that can run more efficiently on CPU. It's a good substitute for demonstrating the capabilities of the larger Qwen models without requiring a powerful GPU.

## Background

Originally, this project was set up to benchmark and train the Qwen2.5-Coder-7B-Instruct model, but that requires significant GPU resources. We've now updated all scripts to use the 1.5B parameter version, which provides a better balance of performance and resource requirements.

## Project Scripts

The project includes several scripts:

1. `run_benchmarks.sh` - Benchmarks the 1.5B model to find optimal configuration
2. `run_optimal_training.sh` - Runs training with the recommended settings
3. `simple_model_demo.py` - Simplified demo for generating code samples
4. `run_model_demo.sh` - Script to run multiple demo examples

## Future Work

To use even larger Qwen models (7B or 13B):

1. Set up a machine with a GPU that has sufficient VRAM (16GB+ recommended)
2. Install CUDA toolkit and appropriate drivers
3. Update the TARGET_MODEL in run_benchmarks.sh to the desired model size
4. Run the benchmarking and training scripts 