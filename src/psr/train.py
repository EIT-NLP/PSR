"""Stage-wise supervised fine-tuning. Launch with `accelerate launch`."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .data import STAGES, TimeSeriesCollator, TimeSeriesJSONL
from .runtime import build_tokenizer_and_model, load_checkpoint, load_config, project_path, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one capability stage")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--resume-from", help="Previous stage checkpoint directory or model.pt")
    parser.add_argument("--output", help="Checkpoint directory; defaults to outputs/ARCH/STAGE")
    parser.add_argument("--max-steps", type=int, help="Optional short run for smoke checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    set_seed(int(training.get("seed", 2026)))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 4)),
        mixed_precision=training.get("precision", "bf16"),
    )
    if args.stage != "perceive" and not args.resume_from:
        raise ValueError("Solve/Reason training needs --resume-from for the previous stage")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    tokenizer, model = build_tokenizer_and_model(config)
    if args.resume_from:
        previous_stage = load_checkpoint(model, args.resume_from, config)
        expected = {"perceive": None, "solve": "perceive", "reason": "solve"}[args.stage]
        if previous_stage != expected:
            raise ValueError(f"Stage {args.stage} requires a {expected} checkpoint, got {previous_stage}")
    model.set_stage(args.stage)
    channels = int(config["model"]["channels"])
    dataset = TimeSeriesJSONL(project_path(config["data"][args.stage]), channels, args.stage)
    collator = TimeSeriesCollator(tokenizer, channels, int(training.get("max_text_tokens", 1024)))
    loader = DataLoader(dataset, batch_size=int(training.get("batch_size_per_gpu", 4)),
                        shuffle=True, collate_fn=collator, num_workers=int(training.get("num_workers", 0)))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(training.get("learning_rate", 2e-5)),
                                  weight_decay=float(training.get("weight_decay", 0.01)))
    epochs = int(training.get("epochs", {}).get(args.stage, 1))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    accumulation = int(training.get("gradient_accumulation_steps", 4))
    total_steps = math.ceil(len(loader) / accumulation) * epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * float(training.get("warmup_ratio", 0.0)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    model.train()
    completed_steps = 0
    for epoch in range(epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, float(training.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                completed_steps += 1
                if completed_steps % int(training.get("log_every", 10)) == 0:
                    accelerator.print(f"stage={args.stage} epoch={epoch + 1} step={completed_steps} loss={loss.item():.4f}")
            if args.max_steps and completed_steps >= args.max_steps:
                break
        if args.max_steps and completed_steps >= args.max_steps:
            break
    output = project_path(args.output) if args.output else project_path(
        Path("outputs") / config["model"]["architecture"] / args.stage
    )
    save_checkpoint(accelerator, model, output, config, args.stage)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output / "tokenizer")
        accelerator.print(f"Saved reference checkpoint to {output}")


if __name__ == "__main__":
    main()
