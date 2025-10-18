#!/usr/bin/env python3
"""
Simple script to extract Lean 4 data for fine-tuning
"""

import os
import re
import json
from pathlib import Path
import subprocess
import sys

def install_lean():
    """Install Lean 4 if not available"""
    try:
        result = subprocess.run(['lean', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Lean already installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("Lean 4 not found. Please install it from: https://leanprover-community.github.io/get_started.html")
    print("Or use: curl -L https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh")
    return False

def extract_theorems_from_lean_file(file_path):
    """Extract theorems and proofs from a single .lean file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    theorems = []
    
    # Regex patterns for Lean 4 syntax
    # Use non-capturing group in the lookahead to avoid extra tuple elements
    theorem_pattern = (
        r'(theorem|lemma|example)'           # kind
        r'\s+([^:]+?)'                      # name and params (up to colon)
        r'\s*:\s*'                         # separator
        r'(.+?)'                             # statement (non-greedy)
        r'\s*:=\s*'                        # definition separator
        r'(.+?)'                             # proof body (non-greedy)
        r'(?=\n\s*(?:theorem|lemma|example|end|$))'  # next decl/end or EOF
    )
    
    matches = re.findall(theorem_pattern, content, re.DOTALL | re.MULTILINE)
    
    for match in matches:
        theorem_type, name, statement, proof = match
        proof = proof.strip()
        
        # Clean up the proof (remove extra whitespace, comments)
        proof_lines = []
        for line in proof.split('\n'):
            line = line.strip()
            if line and not line.startswith('--'):  # Skip empty lines and comments
                proof_lines.append(line)
        
        if proof_lines:
            theorems.append({
                'type': theorem_type.strip(),
                'name': name.strip(),
                'statement': statement.strip(),
                'proof': '\n'.join(proof_lines),
                'file': str(file_path)
            })
    
    return theorems

def process_directory(directory_path):
    """Process all .lean files in a directory"""
    all_theorems = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.lean'):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                theorems = extract_theorems_from_lean_file(file_path)
                all_theorems.extend(theorems)
    
    return all_theorems

def create_training_data(theorems):
    """Convert theorems to training format"""
    training_examples = []
    
    for theorem in theorems:
        # Create context-proof pairs
        context = f"### CONTEXT\n{theorem['type']} {theorem['name']} : {theorem['statement']} :="
        proof = f"### PROOF\n{theorem['proof']}"
        
        # Combine for next-token prediction
        full_text = f"{context}\n{proof}"
        
        training_examples.append({
            'text': full_text,
            'context': context,
            'proof': proof,
            'type': theorem['type'],
            'name': theorem['name'],
            'file': theorem['file']
        })
    
    return training_examples

def main():
    """Main function to extract Lean data"""
    print("Lean 4 Data Extraction Script")
    print("=" * 40)
    
    # Optional: skip Lean binary check when only parsing text
    skip_check = False
    if "--no-lean-check" in sys.argv:
        skip_check = True
        sys.argv.remove("--no-lean-check")
    if not skip_check:
        # Check if Lean is installed (useful if later adding verification)
        if not install_lean():
            return
    
    # Get directory to process
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input("Enter path to directory containing .lean files: ").strip()
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return
    
    print(f"Processing directory: {directory}")
    
    # Extract theorems
    theorems = process_directory(directory)
    print(f"Found {len(theorems)} theorems")
    
    if not theorems:
        print("No theorems found!")
        return
    
    # Convert to training format
    training_data = create_training_data(theorems)
    
    # Save results
    output_file = "lean_training_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(training_data)} training examples to {output_file}")
    
    # Show some examples
    print("\nSample training examples:")
    for i, example in enumerate(training_data[:3]):
        print(f"\n--- Example {i+1} ---")
        print(example['text'][:200] + "..." if len(example['text']) > 200 else example['text'])
    
    # Statistics
    print(f"\nStatistics:")
    print(f"Total examples: {len(training_data)}")
    print(f"Average proof length: {sum(len(ex['proof']) for ex in training_data) / len(training_data):.1f} characters")
    
    # Save a simple text file for easy inspection
    with open("lean_examples.txt", 'w', encoding='utf-8') as f:
        for example in training_data[:10]:  # First 10 examples
            f.write(f"{example['text']}\n\n{'='*50}\n\n")

if __name__ == "__main__":
    main()

