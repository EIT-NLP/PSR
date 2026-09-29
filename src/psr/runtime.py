"""Configuration, backbone loading, and portable reference checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import build_model

STAGES = ("perceive", "solve", "reason")


def training_history(stage: str, previous: list[str] | None = None) -> list[str]:
    """Allow ordered subsets such as Solve only or Perceive → Reason."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    previous = list(previous or [])
    if any(item not in STAGES for item in previous):
        raise ValueError(f"Invalid previous stage history: {previous}")
    if previous and STAGES.index(stage) <= STAGES.index(previous[-1]):
        raise ValueError(f"Stage {stage} must come after {previous[-1]}")
    return previous + [stage]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "model" not in config or "training" not in config or "data" not in config:
        raise ValueError("Config must contain model, training, and data mappings")
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def build_tokenizer_and_model(config: dict[str, Any]) -> tuple[Any, torch.nn.Module]:
    name = config["model"]["backbone"]
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    precision = config["training"].get("precision", "bf16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[precision]
    llm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, trust_remote_code=False)
    return tokenizer, build_model(config, llm).to(dtype=dtype)


def load_checkpoint(model: torch.nn.Module, path: str | Path, config: dict[str, Any]) -> list[str]:
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = config["model"]["architecture"]
    backbone = config["model"]["backbone"]
    if checkpoint["architecture"] != architecture or checkpoint["backbone"] != backbone:
        raise ValueError("Checkpoint architecture/backbone does not match configuration")
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if unexpected or (architecture == "post_align" and missing) or (
        architecture == "pre_align" and any(not key.startswith("llm.") for key in missing)
    ):
        raise ValueError(f"Incompatible checkpoint: missing={missing[:5]}, unexpected={unexpected[:5]}")
    return list(checkpoint.get("history", [checkpoint["stage"]]))


def save_checkpoint(accelerator: Any, model: torch.nn.Module, output: Path,
                    config: dict[str, Any], history: list[str]) -> None:
    state = accelerator.get_state_dict(model)
    architecture = config["model"]["architecture"]
    if architecture == "pre_align":
        state = {key: tensor for key, tensor in state.items() if not key.startswith("llm.")}
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        accelerator.save({
            "architecture": architecture,
            "backbone": config["model"]["backbone"],
            "stage": history[-1],
            "history": history,
            "state_dict": state,
        }, output / "model.pt")
    accelerator.wait_for_everyone()
