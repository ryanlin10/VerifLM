"""Utilities to download and cache Lean theorem sources."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SourceSpec:
    """Metadata needed to fetch a Lean data source."""

    name: str
    url: str
    dest_subdir: str
    license: str
    description: str
    branch: Optional[str] = None
    shallow: bool = True


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "leanft"


AVAILABLE_SOURCES: Mapping[str, SourceSpec] = {
    "mathlib4": SourceSpec(
        name="mathlib4",
        url="https://github.com/leanprover-community/mathlib4",
        dest_subdir="mathlib4",
        license="Apache-2.0",
        description="Community-driven Lean 4 mathematics library (mathlib4).",
    ),
    "stdlib": SourceSpec(
        name="stdlib",
        url="https://github.com/leanprover/lean4",
        dest_subdir="lean4",
        license="Apache-2.0",
        description="Lean 4 core repository containing the stdlib under src/.",
    ),
    "mini_f2f": SourceSpec(
        name="mini_f2f",
        url="https://github.com/openai/miniF2F",
        dest_subdir="miniF2F",
        license="MIT",
        description="MiniF2F benchmark containing Lean formalizations of contest problems.",
    ),
    # LeanDojo support can be toggled on once API stability is confirmed.
    "leandojo": SourceSpec(
        name="leandojo",
        url="https://github.com/lean-dojo/LeanDojo",
        dest_subdir="LeanDojo",
        license="Apache-2.0",
        description="LeanDojo dataset utilities (requires API access; optional).",
        shallow=False,
    ),
}


class SourceNotFoundError(ValueError):
    """Raised when a requested data source name is unknown."""


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_git_command(args: List[str], cwd: Optional[Path] = None) -> None:
    git_path = shutil.which("git")
    if git_path is None:
        raise RuntimeError("git is required to fetch Lean sources but was not found on PATH.")
    cmd = [git_path] + args
    LOGGER.debug("Executing git command: %s (cwd=%s)", " ".join(cmd), cwd)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _clone_repo(spec: SourceSpec, destination: Path) -> None:
    _ensure_directory(destination.parent)
    args = ["clone", spec.url, str(destination)]
    if spec.branch:
        args.extend(["--branch", spec.branch])
    if spec.shallow:
        args.extend(["--depth", "1"])
    LOGGER.info("Cloning %s into %s", spec.name, destination)
    _run_git_command(args)


def _update_repo(spec: SourceSpec, destination: Path) -> None:
    LOGGER.info("Updating %s in %s", spec.name, destination)
    fetch_args = ["fetch", "--all"]
    pull_args = ["pull"]
    if spec.shallow:
        fetch_args.extend(["--depth", "1"])
    _run_git_command(fetch_args, cwd=destination)
    _run_git_command(pull_args, cwd=destination)


def fetch_source(spec: SourceSpec, root: Path) -> Path:
    """Ensure a source repository is present under ``root`` and return its path."""

    destination = root / spec.dest_subdir
    if destination.exists():
        LOGGER.info("Skipping fetch for %s; directory already exists.", spec.name)
        return destination
    _clone_repo(spec, destination)
    return destination


def fetch_sources(
    dest: Path,
    names: Iterable[str],
    allow_missing: bool = False,
) -> Dict[str, Path]:
    """Fetch multiple Lean sources into ``dest`` and return their paths."""

    dest = dest.expanduser().resolve()
    _ensure_directory(dest)
    resolved: Dict[str, Path] = {}
    for name in names:
        key = name.lower()
        if key not in AVAILABLE_SOURCES:
            if allow_missing:
                LOGGER.warning("Unknown source '%s'; skipping.", name)
                continue
            raise SourceNotFoundError(f"Unknown source '{name}'. Known sources: {list(AVAILABLE_SOURCES)}")
        spec = AVAILABLE_SOURCES[key]
        try:
            resolved[key] = fetch_source(spec, dest)
        except subprocess.CalledProcessError as exc:
            LOGGER.error("Failed to fetch %s: %s", spec.name, exc)
            raise
    return resolved


def ensure_sources_up_to_date(dest: Path, names: Iterable[str]) -> None:
    """Update repositories for the selected sources if they already exist."""

    for name in names:
        key = name.lower()
        spec = AVAILABLE_SOURCES.get(key)
        if not spec:
            raise SourceNotFoundError(f"Unknown source '{name}'.")
        repo_dir = dest / spec.dest_subdir
        if not repo_dir.exists():
            LOGGER.warning("Repository for %s missing; cloning anew.", name)
            fetch_source(spec, dest)
            continue
        try:
            _update_repo(spec, repo_dir)
        except subprocess.CalledProcessError as exc:
            LOGGER.error("Failed to update %s: %s", spec.name, exc)
            raise


def list_available_sources() -> List[SourceSpec]:
    """Return source specs for documentation or inspection."""

    return list(AVAILABLE_SOURCES.values())
