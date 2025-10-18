#!/usr/bin/env python3
"""
Simple GPT-2 test script without interactive input
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_gpt2():
    """Test GPT-2 model loading and generation"""
    print("Loading GPT-2 model...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Force CPU to avoid MPS memory issues
    device = torch.device("cpu")
    model.to(device)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on device: {device}")
    print(f"Model has {model.num_parameters():,} parameters")
    
    # Test different types of prompts
    test_prompts = [
        "The spider is",  # Completion
        "Question: What is a spider?\nAnswer:",  # Q&A format
        "The following is a conversation:\nHuman: What is a spider?\nAI:",  # Conversational
        "In a formal mathematical proof, we need to show that"  # Academic
    ]
    
    for i, test_prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Test prompt: '{test_prompt}'")
        
        # Tokenize input
        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        print("Generating response...")
        
        # Generate response with better parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                temperature=0.8,  # Higher for more creativity
                top_p=0.9,
                top_k=50,  # Add top_k sampling
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Higher to reduce repetition
                no_repeat_ngram_size=3
            )
        
        # Decode only the new tokens (response)
        new_tokens = outputs[0][len(input_ids[0]):]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"Generated response: {response.strip()}")
    
    print("\n✅ GPT-2 is working correctly!")
    
    return True

if __name__ == "__main__":
    test_gpt2()
