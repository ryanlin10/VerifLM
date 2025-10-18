# 🚀 GPU Desktop Training Guide

## **Step 1: Transfer Files to GPU Desktop**

### Option A: Using `rsync` (Recommended)
```bash
# From your current machine, transfer the entire VerifLM directory
rsync -avz --progress VerifLM/ username@gpu-desktop-ip:/path/to/destination/

# Example:
rsync -avz --progress VerifLM/ user@192.168.1.100:~/VerifLM/
```

### Option B: Using `scp`
```bash
# Transfer the entire directory
scp -r VerifLM/ username@gpu-desktop-ip:/path/to/destination/
```

### Option C: Using Git (if you have a repo)
```bash
# Push to GitHub/GitLab from current machine
git add .
git commit -m "Ready for GPU training"
git push

# Pull on GPU desktop
git clone <your-repo-url>
cd VerifLM
```

## **Step 2: SSH into GPU Desktop**

```bash
ssh username@gpu-desktop-ip
cd VerifLM
```

## **Step 3: Run GPU Setup**

```bash
# Make setup script executable and run it
chmod +x setup_gpu.sh
./setup_gpu.sh
```

## **Step 4: Start Training**

```bash
# Activate environment
conda activate leanft

# Start training with GPU config
python scripts/train_lora.py --config configs/gpu_config.yaml
```

## **Step 5: Monitor Training**

### Option A: Weights & Biases (Recommended)
```bash
# Login to wandb (first time only)
wandb login

# Training will automatically log to wandb
# Check: https://wandb.ai/your-username/leanft
```

### Option B: TensorBoard
```bash
# In another terminal
tensorboard --logdir logs/gpu-training --port 6006

# Open browser to: http://gpu-desktop-ip:6006
```

## **GPU Optimizations Already Included:**

✅ **Mixed Precision Training** (`bf16`) - 50% memory reduction  
✅ **Automatic GPU Placement** (`device_map: auto`)  
✅ **Gradient Accumulation** - Simulates larger batch sizes  
✅ **LoRA Fine-tuning** - Only trains adapter weights  
✅ **Multi-worker Data Loading** - Faster data processing  
✅ **Gradient Clipping** - Prevents exploding gradients  
✅ **Early Stopping** - Prevents overfitting  

## **Expected Performance:**

- **Memory Usage**: ~8-12GB VRAM (with `bf16`)
- **Training Speed**: ~2-5 examples/second (depending on GPU)
- **Time to Convergence**: 2-4 hours (depending on dataset size)

## **Troubleshooting:**

### If you get CUDA out of memory:
```yaml
# Reduce batch size in configs/gpu_config.yaml
train_batch_size: 2
eval_batch_size: 4
gradient_accumulation_steps: 8
```

### If training is too slow:
```yaml
# Increase batch size
train_batch_size: 8
eval_batch_size: 16
gradient_accumulation_steps: 2
```

### If you want to resume training:
```bash
python scripts/train_lora.py --config configs/gpu_config.yaml --resume_from_checkpoint checkpoints/gpt2-lora-lean-gpu/checkpoint-XXX
```

## **Quick Test:**

```bash
# Test GPU setup
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"

# Test model loading
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.bfloat16, device_map='auto')
print(f'Model loaded on: {next(model.parameters()).device}')
"
```
