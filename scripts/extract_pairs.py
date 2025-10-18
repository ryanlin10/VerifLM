#!/usr/bin/env python
"""Extract theorem/proof pairs from Lean repositories."""

from __future__ import annotations

import argparse
from pathlib import Path

from leanft.data import AVAILABLE_SOURCES, extract_pairs
from leanft.data.extract import dump_records
from leanft.utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw"), help="Root directory containing source repos.")
    parser.add_argument("--out", type=Path, default=Path("data/pairs.jsonl"), help="Output JSONL file.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(AVAILABLE_SOURCES.keys()),
        help="Names of sources to extract (subset of available sources).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = {}
    for name in args.sources:
        key = name.lower()
        if key not in AVAILABLE_SOURCES:
            LOGGER.warning("Unknown source '%s'; skipping.", name)
            continue
        path = args.src / AVAILABLE_SOURCES[key].dest_subdir
        if not path.exists():
            LOGGER.warning("Source directory %s missing; run fetch_data first.", path)
            continue
        mapping[key] = path
    if not mapping:
        raise SystemExit("No valid sources found to extract.")
    records = extract_pairs(mapping)
    dump_records(records, args.out)
    LOGGER.info("Wrote %d records to %s", len(records), args.out)


if __name__ == "__main__":
    main()
