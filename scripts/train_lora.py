#!/usr/bin/env python
"""Fine-tune GPT-2 on Lean data using LoRA adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk

from leanft.data import DatasetConfig, tokenize_dataset
from leanft.train import load_lora_model, load_training_config, train
from leanft.utils import initialize_wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="Training config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    initialize_wandb(project="leanft", enabled=config.use_wandb)
    dataset = load_from_disk(str(Path(config.dataset_path)))
    dataset_config = DatasetConfig(
        tokenizer_name=config.model_name,
        max_length=config.seq_len,
        separator=config.prompt_separator,
    )
    tokenized_dataset, tokenizer = tokenize_dataset(dataset, dataset_config)
    model = load_lora_model(config)
    train(
        model,
        tokenized_dataset,
        tokenizer,
        config,
        output_samples=Path(config.samples_path),
    )


if __name__ == "__main__":
    main()
