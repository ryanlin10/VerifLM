"""Data ingestion utilities for Lean fine-tuning."""

from .augment import augment_example, default_paraphrases
from .dataset import DatasetConfig, build_dataset, tokenize_dataset
from .extract import extract_pairs
from .sources import AVAILABLE_SOURCES, SourceSpec, fetch_sources

__all__ = [
    "augment_example",
    "default_paraphrases",
    "build_dataset",
    "tokenize_dataset",
    "DatasetConfig",
    "extract_pairs",
    "AVAILABLE_SOURCES",
    "SourceSpec",
    "fetch_sources",
]
