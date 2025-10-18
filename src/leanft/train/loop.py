"""Training loop for GPT-2 LoRA fine-tuning."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict
from transformers import (
    DefaultDataCollator,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ..eval.metrics import compute_metrics_fn
from ..utils.logging import get_logger
from .config import TrainerSettings

LOGGER = get_logger(__name__)


def _build_training_arguments(config: TrainerSettings, enable_eval: bool) -> TrainingArguments:
    scheduler = config.scheduler
    optimizer = config.optimizer
    evaluation_strategy = "steps" if enable_eval else "no"
    warmup_ratio = scheduler.warmup_ratio if scheduler.warmup_steps == 0 else 0.0
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        fp16=config.mixed_precision == "fp16",
        bf16=config.mixed_precision == "bf16",
        gradient_checkpointing=False,
        dataloader_num_workers=config.num_workers,
        max_steps=config.max_train_steps or -1,
        # Only try to load the best model when evaluation is enabled; otherwise HF
        # requires eval/save strategies to match and will error.
        load_best_model_at_end=enable_eval,
        max_grad_norm=config.gradient_clip_norm,
        report_to=list(config.report_to),
        seed=config.seed,
        logging_dir=config.logging_dir,
        learning_rate=optimizer.learning_rate,
        weight_decay=optimizer.weight_decay,
        adam_beta1=optimizer.beta1,
        adam_beta2=optimizer.beta2,
        adam_epsilon=optimizer.epsilon,
        lr_scheduler_type=scheduler.name,
        warmup_steps=scheduler.warmup_steps,
        warmup_ratio=warmup_ratio,
        label_names=["labels"],
        dataloader_pin_memory=False,
        optim="adamw_torch",
    )


def train(
    model,
    dataset: DatasetDict,
    tokenizer,
    config: TrainerSettings,
    output_samples: Optional[Path] = None,
):
    """Run supervised fine-tuning with optional evaluation sampling."""

    if config.train_split not in dataset:
        raise ValueError(f"Tokenized dataset missing '{config.train_split}' split.")
    train_dataset = dataset[config.train_split]
    eval_dataset = dataset.get(config.eval_split)
    training_args = _build_training_arguments(config, enable_eval=eval_dataset is not None)
    collator = DefaultDataCollator(return_tensors="pt")
    compute_metrics = partial(compute_metrics_fn, tokenizer=tokenizer) if eval_dataset else None
    callbacks = []
    if config.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    LOGGER.info("Starting training for up to %s epochs.", config.num_epochs)
    trainer.train()
    trainer.save_model(config.output_dir)
    if output_samples and eval_dataset:
        output_samples.parent.mkdir(parents=True, exist_ok=True)
        with output_samples.open("w", encoding="utf-8") as handle, torch.no_grad():
            sample_batch = eval_dataset[:10]
            inputs = sample_batch.get("input") or sample_batch.get("inputs")
            if inputs is None:
                inputs = [
                    tokenizer.decode(ids, skip_special_tokens=True) for ids in sample_batch["input_ids"]
                ]
            for input_text in inputs:
                tokens = tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=config.seq_len // 2,
                )
                generated_ids = trainer.model.generate(
                    tokens["input_ids"].to(trainer.model.device),
                    attention_mask=tokens["attention_mask"].to(trainer.model.device),
                    max_new_tokens=256,
                    do_sample=False,
                )[0]
                generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
                handle.write(f"Input:\n{input_text}\n\nGenerated:\n{generated}\n\n---\n")
    return trainer
