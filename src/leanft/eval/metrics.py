"""Evaluation utilities for Lean fine-tuning."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from transformers import EvalPrediction, PreTrainedTokenizerBase


def perplexity_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.from_numpy(np.asarray(logits)).float()
    labels_tensor = torch.from_numpy(np.asarray(labels)).long()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(
        logits_tensor.view(-1, logits_tensor.size(-1)),
        labels_tensor.view(-1),
    )
    return float(torch.exp(loss).item())


def token_f1(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = torch.from_numpy(np.asarray(logits)).argmax(dim=-1)
    labels_tensor = torch.from_numpy(np.asarray(labels)).long()
    mask = labels_tensor != -100
    if mask.sum() == 0:
        return 0.0
    matches = (preds == labels_tensor) & mask
    tp = matches.sum().item()
    predicted = mask.sum().item()
    target = mask.sum().item()
    if predicted == 0 or target == 0:
        return 0.0
    precision = tp / predicted
    recall = tp / target
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics_fn(eval_pred: EvalPrediction, tokenizer: PreTrainedTokenizerBase) -> Dict[str, float]:
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    _ = tokenizer  # Tokenizer kept for potential post-processing hooks.
    perplexity = perplexity_from_logits(logits, labels)
    f1 = token_f1(logits, labels)
    return {"perplexity": perplexity, "token_f1": f1}
