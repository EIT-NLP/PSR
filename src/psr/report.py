"""Render the manuscript's transcribed CSV results as Markdown tables.

This formats reported numbers. It does not run model inference or reproduce scores.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TABLES = {
    "progressive_capability": ("Stage 2 task solving", "task_solving_stage2.csv"),
    "reasoning_results": ("Five-dimensional reasoning", "reasoning_five_dimensions.csv"),
    "reasoning_path": ("Reasoning QA", "reasoning_qa.csv"),
    "ood_results": ("Out-of-domain MTBench", "mtbench_ood.csv"),
    "perception_generalization": ("Perception generalization", "perception_generalization.csv"),
    "reasoning_data_scaling_detailed": ("Reasoning data scaling", "reasoning_data_scaling.csv"),
    "stage_saturation": ("Stage-wise saturation", "stage_saturation.csv"),
    "optimization_alignment": ("Optimization alignment", "optimization_alignment.csv"),
    "frozen_encoder": ("Frozen temporal encoder ablation", "frozen_encoder.csv"),
    "cate_num_comparison_detailed": ("Detailed temporal perception", "perception_detailed.csv"),
}

PATH_LABELS = {
    "no_finetuning": "No fine-tuning",
    "no_training": "No training",
    "stage1_only": "Stage 1 only",
    "stage2_only": "Stage 2 only",
    "stage3_only": "Stage 3 only",
    "stage1_to_stage2": "Stage 1 → 2",
    "stage1_to_stage3": "Stage 1 → 3",
    "stage2_to_stage3": "Stage 2 → 3",
    "stage1_to_stage2_to_stage3": "Stage 1 → 2 → 3",
    "mixed": "Mixed training",
}


def display_value(column: str, value: str) -> str:
    if column == "training_path":
        return PATH_LABELS.get(value, value)
    return value.replace("|", "\\|")


def display_column(column: str) -> str:
    if column.startswith("ratio_"):
        return column.removeprefix("ratio_") + "% data"
    return (column.replace("_", " ").title()
            .replace("Mse", "MSE").replace("Qa", "QA")
            .replace("Mcqa", "MCQA").replace("Tse", "TSE"))


def render_table(table_id: str, results_dir: Path) -> str:
    if table_id not in TABLES:
        raise ValueError(f"Unknown table: {table_id}")
    title, filename = TABLES[table_id]
    path = results_dir / filename
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    if not columns or not rows:
        raise ValueError(f"Empty result table: {path}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed result table: {path}")
    labels = [display_column(column) for column in columns]
    lines = [f"## {title} (`{table_id}`)", "", "| " + " | ".join(labels) + " |",
             "| " + " | ".join("---" for _ in labels) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(display_value(column, row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Format reported CSV values as Markdown")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--table", choices=TABLES)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, help="Write Markdown to this file instead of stdout")
    args = parser.parse_args()
    selected = TABLES if args.all else [args.table]
    content = "\n\n".join(render_table(table_id, args.results_dir) for table_id in selected) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
