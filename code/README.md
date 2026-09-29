# Reference implementation

The executable package is in [`../src/psr/`](../src/psr/). It is an independently written, paper-inspired implementation for open-source readers. It is **not** the original experiment code, and its outputs should not be presented as a numerical reproduction of the manuscript.

| Module | Responsibility |
| --- | --- |
| [`data.py`](../src/psr/data.py) | Validated JSONL records, prompt formatting, padding, causal labels |
| [`models.py`](../src/psr/models.py) | Patch-based temporal encoder, input-space Pre-align, output-side Post-align |
| [`train.py`](../src/psr/train.py) | Perceive → Solve → Reason supervised training with Accelerate |
| [`evaluate.py`](../src/psr/evaluate.py) | Greedy generation, prediction export, metrics |
| [`runtime.py`](../src/psr/runtime.py) | Configuration, backbone loading, stage checkpoints |

## Data contract

Each UTF-8 JSONL row has an `id`, one of the three `stage` values (`perceive`, `solve`, `reason`), a `task` name, a `series` array of time steps × channels, a natural-language `prompt`, and a target `answer` string:

```json
{"id":"sample-1","stage":"solve","task":"forecasting","series":[[1.0,2.0],[2.0,2.0]],"prompt":"Forecast channel 1. Return a JSON array.","answer":"[3.0]"}
```

The number of channels must match `model.channels` in the YAML config. Each sample needs at least one valid time step. For `forecasting` and `imputation`, answers should be JSON numbers or flat numeric arrays. Other tasks use normalized exact-match accuracy; `reasoning` and `open_qa` additionally report ROUGE-L F1 and corpus BLEU. These reference metrics may differ from the paper's original evaluation scripts.

The files in [`../examples/`](../examples/) are tiny **format examples**. They are unsuitable for scientific training or claims about performance.

## Training

From the repository root, install the package with `python -m pip install -e .` after installing a CUDA-compatible PyTorch build. Relative data and output paths are resolved from the current working directory. Update `configs/pre_align.yaml` or `configs/post_align.yaml` with real dataset paths and model settings. Run one stage at a time:

```bash
accelerate launch -m psr.train --config configs/pre_align.yaml --stage perceive
accelerate launch -m psr.train --config configs/pre_align.yaml --stage solve --resume-from outputs/pre_align/perceive
accelerate launch -m psr.train --config configs/pre_align.yaml --stage reason --resume-from outputs/pre_align/perceive-solve
```

The default checkpoint directories encode their path: `outputs/pre_align/perceive`, then `outputs/pre_align/perceive-solve`, then `outputs/pre_align/perceive-solve-reason`. You can also start directly at `solve` or `reason`, or continue from a `perceive` checkpoint directly to `reason`, to form the ordered ablation paths. Use `configs/post_align.yaml` and `outputs/post_align/...` for Post-align. `--max-steps 1` runs a short pipeline check. The default configs point to tiny example data and Qwen3-4B, so they demonstrate the interface rather than useful training. Post-align jointly optimizes a 4B backbone and may require distributed sharding and substantial GPU memory.

In Pre-align, the LLM remains frozen. The first stage in a path trains the temporal encoder and projection; later stages train the projection with the encoder frozen. In the full path, this means Stage 1 trains the encoder and Stages 2–3 freeze it. In Post-align, the temporal encoder, cross-attention adapter, and LLM are trained together. Checkpoints record the architecture, backbone, and ordered stage history.

## Evaluation

```bash
psr-evaluate --config configs/pre_align.yaml \
  --checkpoint outputs/pre_align/perceive-solve-reason \
  --data examples/reason.jsonl \
  --output outputs/pre_align/reason_predictions.jsonl
```

The evaluator writes predictions and a separate `.metrics.json` summary. Generation currently uses greedy decoding without KV caching, so it favors clarity over throughput. It does not include the manuscript's original dataset converters, prompts, split manifests, checkpoint weights, or benchmark-specific answer parsers. The [evaluation and table index](../docs/EVALUATION.md) maps each transcribed result table to its CSV and corresponding reference entry point.

To format a reported table from its CSV, use `psr-render-table --table progressive_capability`. This only displays manuscript numbers and does not run the evaluator.
