#!/usr/bin/env python3
"""
Test with a more modern, instruction-tuned model
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_modern_model():
    """Test with a more modern model that's better at conversations"""
    print("Loading a more modern model...")
    
    # Try a smaller but more capable model - Microsoft's DialoGPT or similar
    # For this demo, let's try a model that's better at instruction following
    
    try:
        # Try to load a more modern model (this might be large)
        model_name = "microsoft/DialoGPT-medium"  # Better at conversations
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Use CPU
        device = torch.device("cpu")
        model.to(device)
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Couldn't load modern model: {e}")
        print("Falling back to GPT-2 with better prompting...")
        return test_gpt2_with_better_prompts()
    
    print(f"Model loaded on device: {device}")
    print(f"Model has {model.num_parameters():,} parameters")
    
    # Test conversational prompts
    test_prompts = [
        "Human: What is a spider?\nAI:",
        "Human: Explain quantum computing in simple terms.\nAI:",
        "Human: How do you prove that 2+2=4?\nAI:",
        "Human: What is the Lean theorem prover?\nAI:"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input part
        ai_response = response[len(prompt):].strip()
        print(f"AI Response: {ai_response}")
    
    print("\n✅ Modern model test completed!")
    return True

def test_gpt2_with_better_prompts():
    """Test GPT-2 with much better prompting techniques"""
    print("Testing GPT-2 with advanced prompting...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    device = torch.device("cpu")
    model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Advanced prompting techniques
    prompts = [
        # Few-shot prompting
        """Q: What is a cat?
A: A cat is a small domesticated carnivorous mammal.

Q: What is a spider?
A:""",
        
        # Chain of thought
        """Let's think step by step. To understand what a spider is, we need to consider:
1. Its biological classification
2. Its physical characteristics  
3. Its behavior

A spider is""",
        
        # Role-based prompting
        """You are a biology teacher explaining to students. Explain what a spider is in simple terms:

A spider is""",
        
        # Lean-specific
        """In Lean 4, to prove a theorem about natural numbers, we typically use:

theorem example : 2 + 2 = 4 :="""
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Advanced Prompt {i} ---")
        print(f"Prompt: {prompt[:100]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        input_ids = inputs["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=80,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ai_response = response[len(prompt):].strip()
        print(f"Response: {ai_response}")
    
    print("\n✅ Advanced prompting test completed!")
    return True

if __name__ == "__main__":
    test_modern_model()
