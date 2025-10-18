"""LoRA preparation for GPT-style models."""

from __future__ import annotations

from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from .config import LoRASettings, TrainerSettings
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def _resolve_dtype(mixed_precision: Optional[str]) -> Optional[torch.dtype]:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return None


def load_lora_model(config: TrainerSettings):
    """Load the base model and attach LoRA adapters."""

    torch_dtype = _resolve_dtype(config.mixed_precision)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        device_map=config.device_map,
    )
    base_model.requires_grad_(False)
    lora_cfg = _build_lora_config(config.lora)
    model = get_peft_model(base_model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("Trainable parameters: %s / %s (%.2f%%)", trainable, total, 100 * trainable / total)
    return model


def _build_lora_config(settings: LoRASettings) -> LoraConfig:
    return LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        target_modules=list(settings.target_modules),
        lora_dropout=settings.dropout,
        bias=settings.bias,
        task_type=TaskType.CAUSAL_LM,
    )
