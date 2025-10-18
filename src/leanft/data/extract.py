"""Extract (statement, proof) pairs from Lean sources."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

DECLARATION_RE = re.compile(r"^\s*(theorem|lemma|example)\b", flags=re.MULTILINE)


@dataclass
class LeanExample:
    """Structured representation of a Lean declaration."""

    decl_type: str
    name: str
    statement: str
    proof: str
    file: Path
    imports: Sequence[str]
    source: Optional[str] = None

    def to_record(self) -> Dict[str, object]:
        return {
            "type": self.decl_type,
            "name": self.name,
            "statement": self.statement,
            "proof": self.proof,
            "file": str(self.file),
            "imports": list(self.imports),
            "source": self.source,
        }


def _strip_block_comments(text: str) -> str:
    """Remove `/- ... -/` comments while preserving newlines."""

    result: List[str] = []
    depth = 0
    i = 0
    length = len(text)
    while i < length:
        window = text[i : i + 2]
        if window == "/-":
            depth += 1
            i += 2
            continue
        if window == "-/" and depth > 0:
            depth -= 1
            i += 2
            continue
        char = text[i]
        if depth > 0:
            # Preserve line structure for diagnostics.
            result.append("\n" if char == "\n" else " ")
            i += 1
            continue
        result.append(char)
        i += 1
    return "".join(result)


def _strip_line_comments(text: str) -> str:
    """Remove ``--`` comments, keeping trailing newline."""

    cleaned_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        marker = line.find("--")
        if marker >= 0:
            cleaned_lines.append(line[:marker])
        else:
            cleaned_lines.append(line)
    return "".join(cleaned_lines)


def _normalise_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _split_header_statement(header_tail: str) -> str:
    """Return the statement portion from the header tail string."""

    depth = 0
    for idx, char in enumerate(header_tail):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            statement = header_tail[idx + 1 :].strip()
            return statement
    return header_tail.strip()


def _collect_imports(cleaned_source: str) -> List[str]:
    imports: List[str] = []
    for line in cleaned_source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("import "):
            imports.append(stripped.split("import ", 1)[1].strip())
            continue
        # Stop when hitting first non-import to avoid scanning entire file.
        if imports:
            break
    return imports


def _iter_declaration_blocks(content: str) -> Iterable[str]:
    for match in DECLARATION_RE.finditer(content):
        start = match.start()
        next_match = DECLARATION_RE.search(content, match.end())
        end = next_match.start() if next_match else len(content)
        block = content[start:end].strip()
        if ":=" not in block:
            continue
        yield block


def _parse_block(block: str, file_path: Path, imports: Sequence[str], source: Optional[str]) -> Optional[LeanExample]:
    header, proof = block.split(":=", 1)
    header = header.strip()
    proof = _normalise_whitespace(proof.strip())
    header_parts = header.split(None, 2)
    if len(header_parts) < 2:
        LOGGER.debug("Skipping block in %s due to malformed header: %s", file_path, header)
        return None
    decl_type = header_parts[0]
    name = header_parts[1].rstrip(".")
    tail = header_parts[2] if len(header_parts) > 2 else ""
    statement = _split_header_statement(tail)
    statement = _normalise_whitespace(statement)
    return LeanExample(
        decl_type=decl_type,
        name=name,
        statement=statement,
        proof=proof,
        file=file_path,
        imports=tuple(imports),
        source=source,
    )


def extract_from_file(file_path: Path, source: Optional[str] = None) -> List[Dict[str, object]]:
    raw_text = file_path.read_text(encoding="utf-8")
    cleaned = _strip_line_comments(_strip_block_comments(raw_text))
    imports = _collect_imports(cleaned)
    records: List[Dict[str, object]] = []
    for block in _iter_declaration_blocks(cleaned):
        example = _parse_block(block, file_path=file_path, imports=imports, source=source)
        if example:
            records.append(example.to_record())
    return records


def extract_pairs(
    sources: Mapping[str, Path] | Iterable[Path] | Path,
    glob: str = "**/*.lean",
) -> List[Dict[str, object]]:
    """Extract theorem pairs from one or more sources.

    Args:
        sources: Either a mapping of ``source_name`` to root directories, an
            iterable of paths (directories or files), or a single ``Path``.
        glob: Glob expression used when traversing directories.
    """

    if isinstance(sources, Mapping):
        iterable: List[tuple[str, Path]] = [(name, Path(path)) for name, path in sources.items()]
    elif isinstance(sources, Path):
        iterable = [(sources.name, sources)]
    else:
        iterable = [(Path(path).name, Path(path)) for path in sources]

    records: List[Dict[str, object]] = []
    for source_name, root in iterable:
        if root.is_file():
            if root.suffix != ".lean":
                LOGGER.debug("Skipping non-Lean file %s", root)
                continue
            records.extend(extract_from_file(root, source=source_name))
            continue
        for lean_file in sorted(root.glob(glob)):
            if lean_file.is_file():
                records.extend(extract_from_file(lean_file, source=source_name))
    LOGGER.info("Extracted %d theorem pairs.", len(records))
    return records


def dump_records(records: Sequence[Dict[str, object]], output: Path) -> None:
    """Write extracted pairs to ``output`` in JSONL format."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
