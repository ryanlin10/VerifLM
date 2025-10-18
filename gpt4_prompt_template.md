# GPT-4/5 Codex Prompt Template

## **Prompt to Generate Better Fine-Tuning Pipeline:**

```
I need you to create a comprehensive fine-tuning pipeline for training a language model (GPT-2) on Lean 4 theorem prover data. Here are my requirements:

**Objective**: Fine-tune GPT-2 using LoRA (Low-Rank Adaptation) with cross-entropy loss on next-token prediction for Lean 4 formal mathematics data.

**Technical Requirements**:
1. **Data Processing**:
   - Parse .lean files to extract theorem statements and proofs
   - Create training pairs in format: "### CONTEXT\n{theorem}\n### PROOF\n{proof}"
   - Handle Lean 4 syntax properly (special tokens, Unicode, etc.)
   - Implement data cleaning (remove comments, normalize whitespace)
   - Create train/validation/test splits

2. **Model Architecture**:
   - Use GPT-2 as base model
   - Apply LoRA with optimal configuration for mathematical reasoning
   - Support for special Lean tokens (→, ∀, ∃, etc.)
   - Custom tokenizer that understands Lean syntax

3. **Training Setup**:
   - Cross-entropy loss for next-token prediction
   - Proper learning rate scheduling
   - Gradient accumulation for larger effective batch sizes
   - Mixed precision training (fp16)
   - Early stopping and checkpointing
   - Evaluation metrics (perplexity, proof completion accuracy)

4. **Code Structure**:
   - Modular design with separate classes for data loading, model, training
   - Configuration management (YAML/JSON)
   - Logging and monitoring (Weights & Biases or TensorBoard)
   - Reproducible training with seed management

5. **Data Sources**:
   - Support for MiniF2F dataset
   - Support for Mathlib4 extraction
   - Support for custom Lean files

**Expected Output**:
- Complete Python codebase with proper imports
- Requirements.txt with all dependencies
- Configuration files
- README with setup instructions
- Example usage scripts

Please generate production-ready code that I can run immediately.
```

## **Additional Specific Requests**:

### For Data Processing:
```
Create a LeanDataProcessor class that:
- Extracts theorem statements from .lean files
- Parses proof steps and tactics
- Creates training examples with proper context
- Handles Lean 4 Unicode symbols correctly
- Implements data augmentation for proof variations
```

### For Model Architecture:
```
Design a LeanGPT2Model class that:
- Extends GPT-2 with Lean-specific embeddings
- Implements LoRA with optimal rank for mathematical reasoning
- Supports special tokens for Lean syntax
- Includes proof completion head
- Handles variable-length sequences efficiently
```

### For Training:
```
Create a LeanTrainer class that:
- Implements curriculum learning (start with simple proofs)
- Uses contrastive learning for proof correctness
- Includes proof verification as auxiliary task
- Implements proper evaluation on held-out theorem sets
- Supports distributed training
```

## **Sample Data Format**:
```python
training_examples = [
    {
        "context": "theorem add_zero (a : ℕ) : a + 0 = a :=",
        "proof": "by simp",
        "difficulty": "easy",
        "length": 1
    },
    {
        "context": "theorem add_comm (a b : ℕ) : a + b = b + a :=",
        "proof": "by induction a; simp [add_zero, add_succ]",
        "difficulty": "medium", 
        "length": 3
    }
]
```

