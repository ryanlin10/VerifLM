#!/usr/bin/env python
"""Evaluate a fine-tuned Lean LoRA model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments

from leanft.data import DatasetConfig, tokenize_dataset
from leanft.eval import compute_metrics_fn
from leanft.utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to the saved LoRA adapter.")
    parser.add_argument("--base-model", default="gpt2", help="Base model identifier used during training.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset path or split directory.")
    parser.add_argument("--split", default=None, help="Dataset split name (if loading a DatasetDict).")
    parser.add_argument("--max-length", type=int, default=1024, help="Sequence length for tokenization.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size per device.")
    parser.add_argument("--samples", type=Path, default=Path("checkpoints/samples.txt"), help="Sample output file.")
    parser.add_argument("--num-workers", type=int, default=2, help="Evaluation dataloader workers.")
    parser.add_argument("--separator", default="\n\nProof:\n", help="Separator used between input and target during tokenization.")
    return parser.parse_args()


def ensure_tokenized_dataset(
    dataset_or_dict: Dataset | DatasetDict,
    base_model: str,
    max_length: int,
    separator: str,
) -> tuple[Dataset, AutoTokenizer]:
    if isinstance(dataset_or_dict, DatasetDict):
        dataset_dict = dataset_or_dict
    else:
        dataset_dict = DatasetDict({"eval": dataset_or_dict})
    sample_split = next(iter(dataset_dict.values()))
    if "input_ids" in sample_split.column_names:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return sample_split, tokenizer
    cfg = DatasetConfig(tokenizer_name=base_model, max_length=max_length, separator=separator)
    tokenized, tokenizer = tokenize_dataset(dataset_dict, cfg)
    return next(iter(tokenized.values())), tokenizer


def main() -> None:
    args = parse_args()
    dataset_obj = load_from_disk(str(args.data))
    split_name = args.split
    if isinstance(dataset_obj, DatasetDict):
        if split_name and split_name in dataset_obj:
            dataset_obj = dataset_obj[split_name]
        else:
            dataset_obj = dataset_obj.get(split_name or "val") or next(iter(dataset_obj.values()))
    dataset, tokenizer = ensure_tokenized_dataset(dataset_obj, args.base_model, args.max_length, args.separator)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, str(args.model))
    model.eval()
    collator = DefaultDataCollator(return_tensors="pt")

    training_args = TrainingArguments(
        output_dir=str(args.model / "eval"),
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        report_to=[],
        do_train=False,
        do_eval=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics_fn(p, tokenizer=tokenizer),
        data_collator=collator,
    )
    metrics = trainer.evaluate()
    LOGGER.info("Evaluation metrics: %s", metrics)
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    with args.samples.open("w", encoding="utf-8") as handle, torch.no_grad():
        sample_batch = dataset[:10]
        examples = sample_batch.get("input") or sample_batch.get("inputs")
        if examples is None:
            examples = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in sample_batch["input_ids"]
            ]
        for example in examples:
            tokens = tokenizer(
                example,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length // 2,
            )
            generated_ids = model.generate(
                tokens["input_ids"].to(model.device),
                attention_mask=tokens["attention_mask"].to(model.device),
                max_new_tokens=256,
                do_sample=False,
            )[0]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            handle.write(f"Input:\n{example}\n\nGenerated:\n{generated}\n\n---\n")
    LOGGER.info("Sample generations written to %s", args.samples)


if __name__ == "__main__":
    main()
