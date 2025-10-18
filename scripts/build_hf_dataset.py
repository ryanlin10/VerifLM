#!/usr/bin/env python
"""Build a Hugging Face Dataset from augmented Lean data."""

from __future__ import annotations

import argparse
from pathlib import Path

from leanft.data import DatasetConfig, build_dataset
from leanft.utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", type=Path, required=True, help="Augmented JSONL file.")
    parser.add_argument("--out", dest="output_path", type=Path, required=True, help="Output directory for dataset.")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer identifier.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetConfig(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    dataset = build_dataset(args.input_path, cfg)
    args.output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_path))
    LOGGER.info("Dataset saved to %s", args.output_path)


if __name__ == "__main__":
    main()
