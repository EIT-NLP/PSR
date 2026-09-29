"""Single-example inference from a reference checkpoint."""

from __future__ import annotations

import argparse
import json

import torch

from .data import validate_inference_input
from .evaluate import generate_answer
from .runtime import build_tokenizer_and_model, load_checkpoint, load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one time-series question")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="JSON file with series and prompt")
    parser.add_argument("--output", help="Optional JSON output path; defaults to stdout")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    config = load_config(args.config)
    record = json.loads(project_path(args.input).read_text(encoding="utf-8"))
    validate_inference_input(record, int(config["model"]["channels"]))
    tokenizer, model = build_tokenizer_and_model(config)
    load_checkpoint(model, args.checkpoint, config)
    device = torch.device(args.device)
    model.to(device).eval()
    answer = generate_answer(model, tokenizer, record, device, args.max_new_tokens)
    result = {"prompt": record["prompt"], "answer": answer}
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        destination = project_path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
