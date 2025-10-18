"""Dataset utilities for Lean fine-tuning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def _hash_to_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    scaled = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    return min(max(scaled, 0.0), 1.0)


@dataclass
class DatasetConfig:
    tokenizer_name: str = "gpt2"
    max_length: int = 1024
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    separator: str = "\n\nProof:\n"
    pad_to_multiple_of: Optional[int] = 8

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Dataset splits must sum to 1.0; received {total}")


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def split_records(records: Sequence[Dict[str, object]], config: DatasetConfig) -> DatasetDict:
    config.validate()
    splits: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    thresholds = (config.train_ratio, config.train_ratio + config.val_ratio)
    for record in records:
        meta = record.get("meta") or {}
        # Use theorem name as the key so all variants of the same theorem stay together
        name_key = meta.get("name") or record.get("name")
        source_key = meta.get("source") or record.get("source", "")
        if not name_key:
            LOGGER.warning("Record missing name metadata; assigning to train split.")
            splits["train"].append(record)
            continue
        hashed = _hash_to_unit_interval(f"{source_key}:{name_key}")
        if hashed < thresholds[0]:
            splits["train"].append(record)
        elif hashed < thresholds[1]:
            splits["val"].append(record)
        else:
            splits["test"].append(record)
    LOGGER.info(
        "Split %d records into train=%d, val=%d, test=%d",
        len(records),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return DatasetDict({key: Dataset.from_list(items) for key, items in splits.items() if items})


def build_dataset(jsonl_path: Path, config: DatasetConfig) -> DatasetDict:
    records = load_jsonl(jsonl_path)
    if not records:
        raise ValueError(f"No records found in {jsonl_path}")
    return split_records(records, config)


def _prepare_tokenizer(config: DatasetConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _truncate_target(target_ids: List[int], available: int) -> List[int]:
    if available <= 0:
        return target_ids[-1:]
    return target_ids[:available]


def tokenize_dataset(
    dataset: DatasetDict,
    config: DatasetConfig,
) -> tuple[DatasetDict, PreTrainedTokenizerBase]:
    tokenizer = _prepare_tokenizer(config)
    pad_token_id = tokenizer.pad_token_id
    separator_ids = tokenizer.encode(config.separator, add_special_tokens=False)
    eos = tokenizer.eos_token_id

    def _tokenize(example: Dict[str, object]) -> Dict[str, List[int]]:
        input_text: str = example["input"]
        target_text: str = example["target"]
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        max_length = config.max_length
        max_input_tokens = max_length // 2
        if len(input_ids) > max_input_tokens:
            input_ids = input_ids[-max_input_tokens:]
        if eos is not None and (not target_ids or target_ids[-1] != eos):
            target_ids.append(eos)
        available = max_length - (len(input_ids) + len(separator_ids))
        target_ids = _truncate_target(target_ids, max(1, available))
        combined = input_ids + separator_ids + target_ids
        combined = combined[:max_length]
        prefix_len = len(input_ids) + len(separator_ids)
        attention_mask = [1] * len(combined)
        labels = [-100] * prefix_len + target_ids
        labels = labels[: len(combined)]
        if len(labels) < len(combined):
            labels.extend([-100] * (len(combined) - len(labels)))
        if len(combined) < max_length:
            pad_len = max_length - len(combined)
            combined.extend([pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            labels.extend([-100] * pad_len)
        if config.pad_to_multiple_of:
            remainder = len(combined) % config.pad_to_multiple_of
            if remainder:
                extra = config.pad_to_multiple_of - remainder
                combined.extend([pad_token_id] * extra)
                attention_mask.extend([0] * extra)
                labels.extend([-100] * extra)
        return {
            "input_ids": combined,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    try:
        sample_columns = next(iter(dataset.values())).column_names
    except StopIteration as exc:  # pragma: no cover - defensive
        raise ValueError("DatasetDict is empty; cannot tokenize.") from exc
    remove_columns = [col for col in sample_columns if col not in ("input", "target", "meta")]
    tokenized = dataset.map(_tokenize, remove_columns=remove_columns)
    return tokenized, tokenizer
