#!/usr/bin/env python3
"""
Simple LoRA test without datasets dependency
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

def test_lora():
    """Test LoRA functionality"""
    print("Loading tiny GPT-2 model...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    
    # Use CPU to avoid MPS issues
    device = torch.device("cpu")
    model.to(device)
    
    print(f"Model loaded on device: {device}")
    print(f"Original model parameters: {model.num_parameters():,}")
    
    # Apply LoRA configuration
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=4, 
        lora_alpha=16, 
        target_modules=["c_attn"],  # Use c_attn instead of q_proj, v_proj for tiny-gpt2
        bias="none", 
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    print(f"Model with LoRA parameters: {model.num_parameters():,}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test a simple forward pass
    print("Testing forward pass...")
    test_text = "Hello world"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Output shape: {outputs.logits.shape}")
    
    print("✅ LoRA setup is working correctly!")
    return True

if __name__ == "__main__":
    test_lora()
