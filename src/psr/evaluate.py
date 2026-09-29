"""Greedy reference evaluation with per-sample predictions and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import TimeSeriesJSONL, render_prompt
from .metrics import summarize
from .runtime import build_tokenizer_and_model, load_checkpoint, load_config, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a reference checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="JSONL evaluation file")
    parser.add_argument("--output", required=True, help="Prediction JSONL file")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def generate_answer(model: torch.nn.Module, tokenizer: object, record: dict,
                    device: torch.device, max_new_tokens: int) -> str:
    prompt = render_prompt(tokenizer, record["prompt"])
    ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    mask = torch.ones_like(ids)
    series = torch.tensor([record["series"]], dtype=torch.float32, device=device)
    series_mask = torch.ones(series.shape[:2], dtype=torch.bool, device=device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        output = model(series=series, series_mask=series_mask, input_ids=ids, attention_mask=mask)
        next_id = int(output["logits"][0, -1].argmax())
        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=device)], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    config = load_config(args.config)
    tokenizer, model = build_tokenizer_and_model(config)
    load_checkpoint(model, args.checkpoint, config)
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = TimeSeriesJSONL(project_path(args.data), int(config["model"]["channels"]))
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    with output.open("w", encoding="utf-8") as stream:
        for record in dataset:
            prediction = generate_answer(model, tokenizer, record, device, args.max_new_tokens)
            row = {"id": record["id"], "stage": record["stage"], "task": record["task"],
                   "prediction": prediction, "target": record["answer"]}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            predictions.append(row)
    summary = summarize(predictions)
    summary_path = output.with_suffix(".metrics.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Predictions: {output}\nMetrics: {summary_path}")


if __name__ == "__main__":
    main()
