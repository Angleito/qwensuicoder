#!/bin/bash
# Comprehensive script to benchmark and train Qwen2.5-Coder models on PyTorch
# Automatically finds optimal settings and configures Ollama and SLora

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ensure required packages are installed
required_packages=("torch" "torchvision" "numpy" "safetensors" "matplotlib" "psutil" "tqdm" "rich" "requests")
missing_packages=()

for package in "${required_packages[@]}"; do
    python3 -c "import $package" >/dev/null 2>&1 || missing_packages+=("$package")
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "${YELLOW}Missing required packages: ${missing_packages[*]}${NC}"
    echo "Installing missing packages..."
    pip install "${missing_packages[@]}"
fi

# Check for Ollama installation
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Ollama is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install Ollama from: https://ollama.ai/download${NC}"
    exit 1
fi

# Check if Ollama server is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo -e "${YELLOW}Starting Ollama server...${NC}"
    ollama serve &
    sleep 5
fi

# Clear CUDA cache to maximize available memory
echo -e "${GREEN}Clearing CUDA cache to maximize available memory...${NC}"
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
python3 clear_cuda_cache.py --aggressive

# Create output directories
mkdir -p benchmark_results
mkdir -p trained_models
mkdir -p ollama_config

# Display available options
echo -e "\n${GREEN}=== Qwen2.5-Coder Pipeline ===${NC}"
echo "1. Run benchmark to find optimal model"
echo "2. Train model with SLora"
echo "3. Configure Ollama with best model"
echo "4. Run inference with Ollama"
echo "5. Run inference with PyTorch"
echo "6. Run full pipeline (benchmark -> train -> configure)"
echo "7. Exit"

# Prompt for action
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo -e "\n${GREEN}Running benchmark to find optimal settings...${NC}"
        max_size=14
        read -p "Enter maximum model size to test in billions (default: 14): " input_size
        if [[ -n "$input_size" ]]; then
            max_size=$input_size
        fi
        
        python3 model_manager.py --action benchmark --max-model-size $max_size
        ;;
    2)
        echo -e "\n${GREEN}Training model with SLora...${NC}"
        python3 model_manager.py --action train
        ;;
    3)
        echo -e "\n${GREEN}Configuring Ollama with best model...${NC}"
        python3 model_manager.py --action configure-ollama
        ;;
    4)
        echo -e "\n${GREEN}Running inference with Ollama...${NC}"
        read -p "Enter prompt: " prompt
        python3 model_manager.py --action infer-ollama --prompt "$prompt"
        ;;
    5)
        echo -e "\n${GREEN}Running inference with PyTorch...${NC}"
        read -p "Enter prompt: " prompt
        python3 model_manager.py --action infer-pytorch --prompt "$prompt"
        ;;
    6)
        echo -e "\n${GREEN}Running full pipeline...${NC}"
        
        # Ask for max model size
        max_size=14
        read -p "Enter maximum model size to test in billions (default: 14): " input_size
        if [[ -n "$input_size" ]]; then
            max_size=$input_size
        fi
        
        # Run benchmark
        echo -e "\n${GREEN}Step 1: Running benchmark...${NC}"
        python3 model_manager.py --action benchmark --max-model-size $max_size
        
        # Train model
        echo -e "\n${GREEN}Step 2: Training model with SLora...${NC}"
        python3 model_manager.py --action train
        
        # Configure Ollama
        echo -e "\n${GREEN}Step 3: Configuring Ollama...${NC}"
        python3 model_manager.py --action configure-ollama
        
        echo -e "\n${GREEN}Pipeline completed successfully!${NC}"
        echo "You can now use the model for inference with:"
        echo "python3 model_manager.py --action infer-ollama --prompt \"Your prompt here\""
        ;;
    7)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Process completed successfully!${NC}" 