#!/usr/bin/env python
"""Fetch Lean data sources to a local cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from leanft.data import AVAILABLE_SOURCES, fetch_sources
from leanft.utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw"), help="Destination directory for source repos.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(AVAILABLE_SOURCES.keys()),
        help=f"Subset of sources to fetch (default: {list(AVAILABLE_SOURCES)}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved = fetch_sources(args.dest, args.sources)
    for name, path in resolved.items():
        spec = AVAILABLE_SOURCES[name]
        LOGGER.info("%s fetched to %s [%s license]", spec.name, path, spec.license)


if __name__ == "__main__":
    main()
