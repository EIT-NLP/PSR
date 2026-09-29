"""Small, explicit JSONL format and causal-language-model collator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

STAGES = ("perceive", "solve", "reason")


def validate_inference_input(record: dict[str, Any], channels: int) -> None:
    """Validate the fields needed for a single prediction, without a target answer."""
    if not isinstance(record, dict):
        raise ValueError("Inference input must be a JSON object")
    if not isinstance(record.get("prompt"), str) or not record["prompt"]:
        raise ValueError("prompt must be a nonempty string")
    series = record.get("series")
    if not isinstance(series, list) or not series:
        raise ValueError("series must be a nonempty list of time steps")
    for step in series:
        if not isinstance(step, list) or len(step) != channels:
            raise ValueError(f"Each time step must contain {channels} channels")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in step):
            raise ValueError("series values must be finite numbers")


def validate_record(record: dict[str, Any], channels: int, stage: str | None = None) -> None:
    if not isinstance(record, dict):
        raise ValueError("Each JSONL row must be an object")
    required = {"id", "stage", "task", "series", "prompt", "answer"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    if record["stage"] not in STAGES:
        raise ValueError(f"Unknown stage: {record['stage']}")
    if stage and record["stage"] != stage:
        raise ValueError(f"Expected stage {stage}, got {record['stage']}")
    if not all(isinstance(record[key], str) and record[key] for key in ("id", "task", "prompt", "answer")):
        raise ValueError("id, task, prompt, and answer must be nonempty strings")
    validate_inference_input(record, channels)


class TimeSeriesJSONL(Dataset):
    def __init__(self, path: str | Path, channels: int, stage: str | None = None):
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    validate_record(record, channels, stage)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
                self.records.append(record)
        if not self.records:
            raise ValueError(f"Dataset is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def render_prompt(tokenizer: Any, prompt: str) -> str:
    """Use the model's chat format when available; retain plain-text fallback."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"Question: {prompt}\nAnswer:"


class TimeSeriesCollator:
    def __init__(self, tokenizer: Any, channels: int, max_text_tokens: int = 1024):
        self.tokenizer = tokenizer
        self.channels = channels
        self.max_text_tokens = max_text_tokens
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer needs a pad token; set it to EOS if necessary")

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(records)
        max_steps = max(len(item["series"]) for item in records)
        series = torch.zeros(batch_size, max_steps, self.channels, dtype=torch.float32)
        series_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool)
        sequences: list[list[int]] = []
        label_sequences: list[list[int]] = []
        for index, item in enumerate(records):
            length = len(item["series"])
            series[index, :length] = torch.tensor(item["series"], dtype=torch.float32)
            series_mask[index, :length] = True
            prompt_ids = self.tokenizer.encode(render_prompt(self.tokenizer, item["prompt"]), add_special_tokens=False)
            answer_ids = self.tokenizer.encode(item["answer"], add_special_tokens=False)
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                answer_ids.append(eos_id)
            if not answer_ids:
                raise ValueError(f"Answer tokenizes to an empty sequence: {item['id']}")
            if len(prompt_ids) + len(answer_ids) > self.max_text_tokens:
                raise ValueError(f"Sample {item['id']} exceeds max_text_tokens={self.max_text_tokens}")
            sequences.append(prompt_ids + answer_ids)
            label_sequences.append([-100] * len(prompt_ids) + answer_ids)
        max_tokens = max(map(len, sequences))
        input_ids = torch.full((batch_size, max_tokens), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.long)
        labels = torch.full((batch_size, max_tokens), -100, dtype=torch.long)
        for index, (ids, targets) in enumerate(zip(sequences, label_sequences)):
            input_ids[index, : len(ids)] = torch.tensor(ids)
            attention_mask[index, : len(ids)] = 1
            labels[index, : len(targets)] = torch.tensor(targets)
        return {
            "series": series,
            "series_mask": series_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
