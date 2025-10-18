"""Logging helpers with optional Rich integration."""

from __future__ import annotations

import logging
import os
from typing import Optional

try:
    from rich.logging import RichHandler
except ImportError:  # pragma: no cover - Rich optional
    RichHandler = None  # type: ignore


_configured = False


def configure_logging(level: int = logging.INFO, use_rich: Optional[bool] = None) -> None:
    global _configured
    if _configured:
        return
    if use_rich is None:
        use_rich = bool(os.environ.get("LEANFT_USE_RICH", "1") not in {"0", "false", "False"})
    handlers = None
    if use_rich and RichHandler is not None:
        handlers = [RichHandler(rich_tracebacks=True)]
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def initialize_wandb(project: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import wandb
    except ImportError as err:  # pragma: no cover - optional dep
        raise RuntimeError("wandb is not installed but was requested.") from err
    if wandb.run is None:
        wandb.init(project=project)
