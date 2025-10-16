from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    """Load a small but functional GPT-2 model and tokenizer"""
    print("Loading GPT-2 model...")
    # Use the actual GPT-2 small model instead of the broken tiny version
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Force CPU to avoid MPS memory issues
    device = torch.device("cpu")
    print("Using CPU to avoid MPS memory allocation issues")
    
    model.to(device)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer, device

def generate_response(model, tokenizer, device, prompt, max_new_tokens=1000, temperature=0.4, top_p=0.9):
    """Generate a response for the given prompt"""
    # Tokenize input with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2
        )
    
    # Decode only the new tokens (response)
    new_tokens = outputs[0][len(input_ids[0]):]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response.strip()

def main():
    """Interactive testing loop"""
    model, tokenizer, device = load_model()
    
    print("\n=== GPT-2 Interactive Test ===")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for options")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nEnter prompt: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  quit/exit - Exit the program")
                print("  help - Show this help")
                print("  Any other text - Generate response")
                continue
            elif not user_input:
                print("Please enter a prompt.")
                continue
            
            print("Generating response...")
            response = generate_response(model, tokenizer, device, user_input)
            print(f"\nResponse: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
