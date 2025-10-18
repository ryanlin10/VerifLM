#!/usr/bin/env python
"""Expand theorem statements into multiple input variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from leanft.data import augment_example
from leanft.utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", type=Path, required=True, help="Input pairs JSONL.")
    parser.add_argument("--out", dest="output_path", type=Path, required=True, help="Output augmented JSONL.")
    parser.add_argument("--nlx", type=int, default=8, help="Number of natural-language paraphrases to generate.")
    return parser.parse_args()


def read_pairs(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    records = 0
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as out_handle:
        for record in read_pairs(args.input_path):
            augmented = augment_example(record, nl_variants=args.nlx)
            for item in augmented:
                out_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                records += 1
    LOGGER.info("Wrote %d augmented variants to %s", records, args.output_path)


if __name__ == "__main__":
    main()
