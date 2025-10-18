"""Expand Lean theorem statements into multiple input variants."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

DEFAULT_TEMPLATES = [
    "Prove that {statement}",
    "Show that {statement}",
    "Demonstrate that {statement} holds",
    "Establish that {statement}",
    "In ℕ, {statement}",
    "Commutativity property: {statement}",
    "Lean goal: {statement}",
    "Mathematical claim: {statement}",
    "Verify that {statement}",
    "Conclude that {statement}",
]

ParaphraserFn = Callable[[str, int], Sequence[str]]


def default_paraphrases(statement: str, limit: int) -> List[str]:
    phrases: List[str] = []
    for template in DEFAULT_TEMPLATES:
        phrase = template.format(statement=statement)
        if phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= limit:
            break
    return phrases


def _math_form(statement: str) -> str:
    """Return a math-centric rendering, falling back to the original statement."""

    cleaned = statement.strip()
    if "=" in cleaned or "≠" in cleaned:
        return cleaned
    if "≤" in cleaned or "≥" in cleaned:
        return cleaned
    if "∧" in cleaned or "∨" in cleaned:
        return cleaned
    # Fallback to a declarative template when no obvious symbolic hook exists.
    return f"{cleaned}"


def _formal_header(example: Mapping[str, str]) -> str:
    return f"{example['type']} {example['name']} : {example['statement']}"


def _unique(sequence: Iterable[str], max_chars: int) -> List[str]:
    seen = OrderedDict()
    for item in sequence:
        trimmed = item.strip()
        if not trimmed or len(trimmed) > max_chars:
            continue
        if trimmed not in seen:
            seen[trimmed] = None
    return list(seen.keys())


def augment_example(
    example: Mapping[str, str],
    *,
    paraphraser: Optional[ParaphraserFn] = None,
    nl_variants: int = 8,
    max_chars: int = 512,
    include_formal_header: bool = True,
    include_math_form: bool = True,
) -> List[Dict[str, object]]:
    """Expand a theorem record into multiple supervised training pairs."""

    statement = example["statement"]
    proof = example["proof"]
    base_variants: List[str] = []

    if include_formal_header:
        base_variants.append(_formal_header(example))

    if include_math_form:
        base_variants.append(_math_form(statement))

    paraphrases: List[str] = []
    paraphrase_budget = max(nl_variants, 0)
    if paraphraser:
        try:
            paraphrases.extend(paraphraser(statement, paraphrase_budget))
        except Exception as err:  # pragma: no cover - defensive against external APIs
            # Fall back to deterministic templates on failure.
            paraphrase_budget = max(paraphrase_budget, 5)
            paraphrases.extend(default_paraphrases(statement, paraphrase_budget))
    if not paraphrases:
        paraphrases = default_paraphrases(statement, paraphrase_budget or len(DEFAULT_TEMPLATES))

    base_variants.extend(paraphrases)
    unique_variants = _unique(base_variants, max_chars=max_chars)

    meta = {
        "type": example.get("type"),
        "name": example.get("name"),
        "file": example.get("file"),
        "source": example.get("source"),
        "statement": statement,
    }

    augmented: List[Dict[str, object]] = []
    for idx, variant in enumerate(unique_variants):
        augmented.append(
            {
                "input": variant,
                "target": proof,
                "meta": {**meta, "variant_id": idx},
            }
        )
    return augmented
