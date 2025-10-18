"""Configuration schema for Lean fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf


@dataclass
class LoRASettings:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["c_attn"]) 
    bias: str = "none"


@dataclass
class OptimizerSettings:
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


@dataclass
class SchedulerSettings:
    name: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.03


@dataclass
class TrainerSettings:
    model_name: str = "gpt2"
    seq_len: int = 1024
    prompt_separator: str = "\n\nProof:\n"
    dataset_path: str = "data/hf"
    train_split: str = "train"
    eval_split: str = "val"
    samples_path: str = "checkpoints/samples.txt"
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    max_train_steps: Optional[int] = None
    eval_steps: int = 200
    logging_steps: int = 50
    save_steps: int = 200
    early_stopping_patience: Optional[int] = 5
    gradient_clip_norm: float = 1.0
    mixed_precision: Optional[str] = "bf16"
    device_map: Optional[str] = None
    seed: int = 42
    num_workers: int = 4
    output_dir: str = "checkpoints"
    logging_dir: str = "logs"
    report_to: List[str] = field(default_factory=list)
    use_wandb: bool = False
    lora: LoRASettings = field(default_factory=LoRASettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)


def load_training_config(path: Path | str) -> TrainerSettings:
    """Load a training config from YAML/JSON using OmegaConf."""

    loaded = OmegaConf.load(str(path))
    structured = OmegaConf.structured(TrainerSettings)
    merged = OmegaConf.merge(structured, loaded)
    return OmegaConf.to_object(merged)


def save_config(config: TrainerSettings, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conf = OmegaConf.structured(config)
    OmegaConf.save(conf, str(path))
