#!/bin/bash
# GPU Desktop Setup Script for VerifLM Training

echo "🚀 VerifLM GPU Training Setup"
echo "=============================="

# Check if we're on GPU machine
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "❌ No NVIDIA GPU detected. This script is for GPU machines."
    exit 1
fi

# Check Python environment
echo ""
echo "🐍 Checking Python environment..."
if command -v conda &> /dev/null; then
    echo "✅ Conda available"
    # Create environment if it doesn't exist
    if ! conda env list | grep -q "leanft"; then
        echo "Creating conda environment 'leanft'..."
        conda create -n leanft python=3.10 -y
    fi
    echo "Activating conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate leanft
else
    echo "⚠️  Conda not found. Using system Python."
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.43 accelerate>=0.33 peft>=0.11 datasets>=2.20
pip install wandb omegaconf pyyaml

# Check CUDA availability
echo ""
echo "🔍 Checking CUDA availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p checkpoints logs

# Check if data exists
echo ""
echo "📊 Checking training data..."
if [ -d "VerifLM/data/hf" ]; then
    echo "✅ Training data found"
    echo "Dataset size:"
    python -c "
from datasets import load_from_disk
ds = load_from_disk('VerifLM/data/hf')
print(f'Train: {len(ds[\"train\"])} examples')
print(f'Val: {len(ds[\"val\"])} examples')
print(f'Test: {len(ds[\"test\"])} examples')
"
else
    echo "❌ Training data not found. Run data preparation first."
    echo "Run: python scripts/build_hf_dataset.py"
fi

echo ""
echo "🎯 Ready to train! Run:"
echo "python scripts/train_lora.py --config configs/gpu_config.yaml"
echo ""
echo "📊 Monitor training with:"
echo "wandb login  # (if using wandb)"
echo "tensorboard --logdir logs/gpu-training"
