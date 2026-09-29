"""Transparent reference metrics for generated task answers."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any


def normalize_answer(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())


def numeric_values(value: str) -> list[float]:
    """Require a JSON number or numeric array to avoid silent parsing errors."""
    parsed = json.loads(value)
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool) and math.isfinite(parsed):
        return [float(parsed)]
    if isinstance(parsed, list) and parsed and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in parsed):
        return [float(x) for x in parsed]
    raise ValueError("Expected a JSON number or flat numeric array")


def summarize(predictions: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    """Compute per-task metrics, counting invalid numerical output explicitly."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[row["task"]].append(row)
    summary: dict[str, dict[str, float | int | None]] = {}
    for task, rows in groups.items():
        result: dict[str, float | int | None] = {"count": len(rows)}
        if task in {"forecasting", "imputation"}:
            squared_errors = []
            valid = 0
            for row in rows:
                try:
                    target = numeric_values(row["target"])
                    prediction = numeric_values(row["prediction"])
                    if len(target) != len(prediction):
                        continue
                    valid += 1
                    squared_errors.extend((a - b) ** 2 for a, b in zip(target, prediction))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
            result["valid_count"] = valid
            result["mse"] = sum(squared_errors) / len(squared_errors) if squared_errors else None
        else:
            result["accuracy"] = sum(
                normalize_answer(row["prediction"]) == normalize_answer(row["target"]) for row in rows
            ) / len(rows)
            if task in {"reasoning", "open_qa"}:
                from rouge_score import rouge_scorer
                from sacrebleu import corpus_bleu

                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                result["rouge_l_f1"] = sum(
                    scorer.score(row["target"], row["prediction"])["rougeL"].fmeasure for row in rows
                ) / len(rows)
                result["bleu"] = corpus_bleu(
                    [row["prediction"] for row in rows],
                    [[row["target"] for row in rows]],
                ).score
        summary[task] = result
    return summary
