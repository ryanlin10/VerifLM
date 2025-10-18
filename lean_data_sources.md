# Lean 4 Data Sources for Fine-Tuning

## **Available Lean Datasets:**

### 1. **Lean 4 Mathlib** (Best Source)
- **Location**: https://github.com/leanprover-community/mathlib4
- **Size**: Massive - thousands of theorems and proofs
- **Format**: `.lean` files with formal proofs
- **Quality**: Highest quality, well-documented, community-verified

### 2. **Lean 4 Standard Library**
- **Location**: https://github.com/leanprover/lean4/tree/master/src
- **Size**: Core Lean 4 definitions and basic theorems
- **Format**: Official Lean 4 source code

### 3. **MiniF2F Dataset**
- **Location**: https://github.com/openai/miniF2F
- **Size**: ~300 formal mathematics problems
- **Format**: Lean 4 formalizations of competition math problems

### 4. **ProofNet Dataset**
- **Location**: https://github.com/ProofNet-Org/ProofNet
- **Size**: Large collection of formal proofs
- **Format**: Multiple formal proof systems including Lean

## **Data Processing Pipeline:**

### Step 1: Extract Lean Code
```bash
# Clone mathlib4
git clone https://github.com/leanprover-community/mathlib4.git
cd mathlib4

# Extract all .lean files
find . -name "*.lean" -type f > lean_files.txt
```

### Step 2: Clean and Format Data
- Remove comments and documentation
- Extract theorem statements and proofs
- Create training pairs: (context, proof)
- Tokenize with Lean-aware tokenizer

### Step 3: Training Format
```
### CONTEXT
[theorem statement or goal]

### PROOF
[formal proof in Lean]
```

## **Recommended Approach:**
1. Start with MiniF2F (smaller, curated)
2. Scale up to Mathlib4 (massive, comprehensive)
3. Use both theorem statements and proof steps as training data

