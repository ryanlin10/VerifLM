# Lean Fine-Tuning Toolkit

Production-ready pipeline for extracting Lean theorem/proof pairs, augmenting inputs, and fine-tuning GPT-2 with LoRA adapters.

## Highlights
- Fetches Mathlib4, Lean 4 stdlib, MiniF2F (optional LeanDojo) with caching and license annotations.
- Robust Lean parser that captures theorem/lemma/example statements, proofs, imports, and metadata.
- Input augmentation producing formal headers, math-form reductions, and configurable natural-language paraphrases.
- Hugging Face `DatasetDict` builder with leakage-safe file hashing and GPT-2 compatible tokenization/masking.
- LoRA fine-tuning loop using PEFT + Hugging Face Trainer with early stopping, mixed precision, and rich logging.
- Evaluation script reporting perplexity/token F1 and saving qualitative proof generations.

## Quickstart
1. **Create environment** (Python 3.10–3.12):
   ```bash
   conda create -n leanft python=3.11
   conda activate leanft
   ```
   If `pyarrow` fails via pip, install it first with `conda install -c conda-forge pyarrow`.

2. **Install project**:
   ```bash
   make setup
   ```

3. **Run the end-to-end CPU-friendly pipeline on a small subset**:
   ```bash
   make fetch
   make extract
   make augment
   make build
   make train
   ```
   Each command has overridable environment variables (see `Makefile`).

## Workflow
- `scripts/fetch_data.py`: clones Lean sources into `data/raw`, skipping ones already present.
- `scripts/extract_pairs.py`: parses `.lean` files and writes `data/pairs.jsonl`.
- `scripts/augment_inputs.py`: expands each theorem into multiple `(input, target)` variants stored in `data/aug.jsonl`.
- `scripts/build_hf_dataset.py`: splits data with file-level hashing and saves a Hugging Face dataset to `data/hf`.
- `scripts/train_lora.py`: loads `configs/base.yaml`, tokenizes for GPT-2, applies LoRA, and trains.
- `scripts/evaluate.py`: loads saved adapters, reports perplexity/token-F1, and writes generations to `samples.txt`.

## Config & Logging
- `configs/base.yaml` captures LoRA hyperparameters, dataset paths, device map, and logging backends.
- Enable Weights & Biases or TensorBoard by setting `report_to` (e.g., `["wandb"]`) and `use_wandb: true`.
- Mixed precision defaults to `null` for portability; set `bf16` on Apple Silicon/Ampere or `fp16` on CUDA.

## Evaluation Outputs
- `checkpoints/best`: best-performing adapter checkpoint (by perplexity).
- `checkpoints/samples.txt`: proof generations for qualitative inspection.
- Metrics (perplexity, token F1) printed via `scripts/evaluate.py`.

## Testing
- `pytest` covers Lean parsing edge cases and dataset masking behaviour (`tests/`).
- Use `make setup && pytest` before pushing changes or running large jobs.

## Licensing
- See `LICENSES.md` for upstream licenses (Mathlib4, Lean stdlib, MiniF2F, LeanDojo).
