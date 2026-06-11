"""
src/dataset.py — PyTorch Dataset wrapper for the multilingual JSONL files.

Reads train.jsonl / eval.jsonl produced by data/download_datasets.py,
applies Qwen2.5's chat template, and tokenizes with dynamic truncation.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import MAX_SEQ_LENGTH, PROCESSED_DIR


def _load_jsonl(path: Path) -> list[dict]:
    """Loads a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MultilingualChatDataset(Dataset):
    """
    Dataset for Qwen2.5 causal-LM fine-tuning.

    Each item is a dict of:
      input_ids      : tokenized prompt+response (teacher-forced)
      attention_mask : 1 for real tokens, 0 for padding
      labels         : same as input_ids but -100 for the prompt portion
                       (so the loss is computed only on the response)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = MAX_SEQ_LENGTH,
        data_dir: Optional[Path] = None,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.split      = split

        data_dir = data_dir or PROCESSED_DIR
        path = data_dir / f"{split}.jsonl"

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                "Run `python data/download_datasets.py` first."
            )

        raw = _load_jsonl(path)
        self.data = [item for item in raw if self._is_valid(item)]
        print(f"[Dataset] {split}: {len(self.data):,} samples loaded from {path}")

    @staticmethod
    def _is_valid(item: dict) -> bool:
        """Filters out empty or malformed samples."""
        msgs = item.get("messages", [])
        if len(msgs) < 2:
            return False
        # Require at least one user + one assistant turn with real content
        roles = {m["role"] for m in msgs}
        has_content = all(m.get("content", "").strip() for m in msgs)
        return "user" in roles and "assistant" in roles and has_content

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        messages = self.data[idx]["messages"]

        # Apply Qwen2.5's built-in chat template to get the full prompt string
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,   # False = include the assistant response
        )

        # Also build the prompt-only string (everything except assistant response)
        # to compute where labels should be masked
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,    # True = ends with assistant turn marker
        )

        # Tokenize the full text
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids      = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Build labels: -100 for prompt tokens (masked from loss), real ids for response
        prompt_len = len(
            self.tokenizer(
                prompt_text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]
        )

        labels = input_ids.clone()
        labels[:prompt_len] = -100          # mask the prompt
        labels[attention_mask == 0] = -100  # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


def get_collate_fn(tokenizer: PreTrainedTokenizer):
    """
    Dynamic padding collate function — pads each batch to its own max length
    instead of the global max, saving memory during training.
    """
    def collate_fn(batch: list[dict]) -> dict:
        input_ids      = [item["input_ids"]      for item in batch]
        attention_mask = [item["attention_mask"]  for item in batch]
        labels         = [item["labels"]          for item in batch]

        # Pad to longest in this batch
        pad_id = tokenizer.pad_token_id
        max_len = max(ids.size(0) for ids in input_ids)

        def pad(seq, pad_value):
            diff = max_len - seq.size(0)
            return torch.cat([seq, torch.full((diff,), pad_value, dtype=seq.dtype)])

        return {
            "input_ids":      torch.stack([pad(ids, pad_id) for ids in input_ids]),
            "attention_mask": torch.stack([pad(m,   0)      for m in attention_mask]),
            "labels":         torch.stack([pad(lb, -100)    for lb in labels]),
        }

    return collate_fn
