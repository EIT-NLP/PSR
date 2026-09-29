import json
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from psr.data import TimeSeriesCollator, TimeSeriesJSONL
from psr.metrics import summarize
from psr.models import PostAlignModel, PreAlignModel, TemporalPatchEncoder
from psr.report import render_table
from psr.runtime import training_history


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(character) % 27 for character in text]


def test_dataset_and_collator_mask_variable_lengths(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [
        {"id": "a", "stage": "perceive", "task": "pattern", "series": [[1, 2], [3, 4]], "prompt": "P", "answer": "A"},
        {"id": "b", "stage": "perceive", "task": "pattern", "series": [[5, 6]], "prompt": "Q", "answer": "B"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    batch = TimeSeriesCollator(TinyTokenizer(), 2)(list(TimeSeriesJSONL(path, 2, "perceive")))
    assert batch["series"].shape == (2, 2, 2)
    assert batch["series_mask"].tolist() == [[True, True], [True, False]]
    assert batch["labels"][0, 0].item() == -100
    with pytest.raises(ValueError, match="Expected stage"):
        TimeSeriesJSONL(path, 2, "solve")


@pytest.mark.parametrize("architecture", ["pre", "post"])
def test_models_forward_and_stage_policy(architecture):
    llm = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=16, n_layer=1, n_head=2))
    encoder = TemporalPatchEncoder(channels=2, hidden_size=16, patch_size=2, layers=1, heads=2)
    model = PreAlignModel(llm, encoder, 16) if architecture == "pre" else PostAlignModel(llm, encoder, 16, heads=2)
    model.set_stage("solve")
    if architecture == "pre":
        assert not any(parameter.requires_grad for parameter in model.temporal_encoder.parameters())
        assert not any(parameter.requires_grad for parameter in model.llm.parameters())
    else:
        assert all(parameter.requires_grad for parameter in model.parameters())
    output = model(
        series=torch.tensor([[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]]]),
        series_mask=torch.tensor([[True, True, True]]),
        input_ids=torch.tensor([[2, 3, 4, 5]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        labels=torch.tensor([[-100, -100, 4, 5]]),
    )
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.aligner[0].weight.grad is not None if architecture == "pre" else model.cross_attention.in_proj_weight.grad is not None


def test_numeric_metric_reports_invalid_predictions():
    summary = summarize([
        {"task": "forecasting", "prediction": "[1, 3]", "target": "[1, 1]"},
        {"task": "forecasting", "prediction": "invalid", "target": "[1, 1]"},
    ])
    assert summary["forecasting"] == {"count": 2, "valid_count": 1, "mse": 2.0}


def test_ordered_ablation_paths_and_report_renderer():
    assert training_history("reason") == ["reason"]
    assert training_history("reason", ["perceive"]) == ["perceive", "reason"]
    assert training_history("reason", ["perceive", "solve"]) == ["perceive", "solve", "reason"]
    with pytest.raises(ValueError, match="must come after"):
        training_history("perceive", ["solve"])
    results = Path(__file__).resolve().parents[1] / "results"
    markdown = render_table("progressive_capability", results)
    assert "1.328" in markdown and "Stage 2 task solving" in markdown
