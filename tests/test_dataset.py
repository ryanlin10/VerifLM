from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset, DatasetDict

from leanft.data.dataset import DatasetConfig, tokenize_dataset


@dataclass
class DummyTokenizer:
    pad_token: str = "<pad>"
    eos_token: str = "</s>"

    def __post_init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"
        self._vocab: Dict[str, int] = {self.pad_token: 0, self.eos_token: 1}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = text.split()
        ids: List[int] = []
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
            ids.append(self._vocab[token])
        return ids

    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = False, max_length: int | None = None):
        ids = self.encode(text)
        attention = [1] * len(ids)
        tensor_kwargs = {"dtype": torch.long}
        return {
            "input_ids": torch.tensor([ids], **tensor_kwargs),
            "attention_mask": torch.tensor([attention], **tensor_kwargs),
        }


def test_tokenize_dataset_masks_input(monkeypatch) -> None:
    def fake_from_pretrained(name: str):
        return DummyTokenizer()

    monkeypatch.setattr("leanft.data.dataset.AutoTokenizer", type("T", (), {"from_pretrained": staticmethod(fake_from_pretrained)}))

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "input": "theorem foo : a = a",
                        "target": "by rfl",
                        "meta": {"file": "foo.lean", "source": "mathlib4"},
                    }
                ]
            )
        }
    )
    cfg = DatasetConfig(tokenizer_name="dummy", max_length=16, separator=" PROOF ")
    tokenized, tokenizer = tokenize_dataset(dataset, cfg)
    example = tokenized["train"][0]

    # Input tokens should be masked out of the loss.
    labels = example["labels"]
    prefix_len = len(tokenizer.encode("theorem foo : a = a")) + len(tokenizer.encode(" PROOF "))
    assert labels[:prefix_len] == [-100] * prefix_len
    # Target portion must contain at least one non-masked token.
    assert any(label != -100 for label in labels[prefix_len:])
