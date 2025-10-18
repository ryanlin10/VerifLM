from __future__ import annotations

from pathlib import Path

from leanft.data.extract import extract_from_file


def write_lean_file(tmp_path: Path) -> Path:
    content = """
import Mathlib

/-
  Sample block comment that should be ignored.
-/

-- doc comment
theorem add_comm (a b : ℕ) :
    a + b = b + a := by
  -- proof comment
  simpa [Nat.add_comm]

lemma mul_base {n : ℕ} :
    n * 0 = 0 := by
  simpa using Nat.mul_zero n
"""
    file_path = tmp_path / "sample.lean"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_extract_handles_comments_and_multiline(tmp_path: Path) -> None:
    lean_file = write_lean_file(tmp_path)
    records = extract_from_file(lean_file, source="mathlib4")
    assert len(records) == 2

    first = records[0]
    assert first["type"] == "theorem"
    assert first["name"] == "add_comm"
    assert "a + b = b + a" in first["statement"]
    assert "simpa [Nat.add_comm]" in first["proof"]
    assert first["imports"] == ["Mathlib"]
    assert first["source"] == "mathlib4"

    second = records[1]
    assert second["type"] == "lemma"
    assert second["name"] == "mul_base"
    assert "n * 0 = 0" in second["statement"]
    assert "Nat.mul_zero" in second["proof"]
