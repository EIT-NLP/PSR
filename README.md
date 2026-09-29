# Perceive, Solve, Reason

### Progressive Capability Formation in Time Series-LLMs

<p align="center">
  <a href="PAPER_URL_TODO">Paper</a> ·
  <a href="code/README.md">Code</a> ·
  <a href="docs/RESULTS.md">Results</a> ·
  <a href="docs/REPRODUCIBILITY.md">Reproducibility</a>
</p>

> **Release status:** This repository includes a paper-inspired **reference implementation**, a method overview, figures, experimental settings, and reported results. The reference code is independently written; original experiment code, datasets, and checkpoints are not included. `PAPER_URL_TODO` will be replaced when the paper has a public URL.

<p align="center"><img src="assets/architecture.png" alt="Time series textualization, Pre-align, and Post-align architectures" width="820"></p>

## Overview

How do time series large language models acquire the abilities to understand signals, solve tasks, and reason about them? We study a three-stage path:

**Perceive → Solve → Reason**

1. **Perceive:** identify trends, seasonality, anomalies, local patterns, and relationships between variables.
2. **Solve:** tackle forecasting, imputation, anomaly detection, and classification.
3. **Reason:** answer questions that require temporal evidence, inference, extrapolation, and decisions.

We compare this ordered training path with isolated stages and mixed supervision using **Pre-align** and **Post-align** architectures. The paper finds that earlier capabilities support later ones, with the clearest reasoning gains when reasoning supervision is limited. It also studies input format, training duration, optimization behavior, and how capabilities transfer across network layers.

## Highlights

- The full three-stage path delivers the strongest overall reasoning results across the two alignment architectures in the paper's controlled comparisons.
- Ordered training generally improves reasoning over mixed training, especially with less Stage 3 data.
- Explicit temporal alignment yields clearer cross-stage gains than direct time series textualization.
- The resulting models generalize well on out-of-domain weather and finance tasks in MTBench.

## Repository map

```text
.
├── assets/                 # GitHub preview image
├── .github/workflows/      # Reference code checks
├── code/                   # Implementation guide and data contract
├── configs/                # Pre-align and Post-align configurations
├── docs/
│   ├── DATA.md             # Dataset sources and stage mapping
│   ├── REPRODUCIBILITY.md  # Hardware, optimizer, training, evaluation
│   └── RESULTS.md          # Main findings and metrics
├── environment.example.yml # Starting Conda environment
├── examples/               # Tiny JSONL format examples
├── results/                # Machine-readable tables transcribed from paper
├── src/psr/                # Models, training, and evaluation
└── tests/                  # Small reference implementation checks
```

## Environment

The manuscript reports **2 × NVIDIA A800**, **Qwen3-4B** as the primary backbone, PyTorch, Hugging Face Transformers, DeepSpeed, Accelerate, and **bfloat16**. Qwen3-1.7B is used for supplementary experiments. This reference package targets Python 3.10+ and a modern Transformers version with Qwen3 support. Install a CUDA-compatible PyTorch build for your machine, then install the package:

```bash
conda env create -f environment.example.yml
conda activate perceive-solve-reason
# Install the appropriate CUDA-enabled PyTorch build for your system.
python -m pip install -e .
```

The [reported experiment settings](docs/REPRODUCIBILITY.md) are documented separately from these reference package dependencies.

## Training and evaluation code

The [implementation guide](code/README.md) describes the JSONL input format, model components, staged training commands, and evaluation. The two fusion paths use a patch-based time series encoder: **Pre-align** projects temporal tokens into the LLM input embedding space; **Post-align** uses cross-attention from LLM hidden states to temporal patches before the output head.

```bash
accelerate launch -m psr.train --config configs/pre_align.yaml --stage perceive
accelerate launch -m psr.train --config configs/pre_align.yaml --stage solve --resume-from outputs/pre_align/perceive
accelerate launch -m psr.train --config configs/pre_align.yaml --stage reason --resume-from outputs/pre_align/solve
psr-evaluate --config configs/pre_align.yaml --checkpoint outputs/pre_align/reason --data examples/reason.jsonl --output outputs/pre_align/predictions.jsonl
```

The included data are tiny format examples. Replace them with authorized research data before meaningful training. The code has not reproduced the manuscript's numerical results.

## Results at a glance

On out-of-domain **MTBench**, the Pre-align model scores **67.24** on Weather MCQA and **68.19** on 7-day Finance 3-way classification. The Post-align model reaches **0.160 MSE** on 7-day Finance forecasting and **0.744 MSE** on 30-day forecasting. Lower MSE is better; higher accuracy is better. See [all paper results](docs/RESULTS.md) and [CSV tables](results/).

## Paper

The paper URL will be added here when it becomes public: `PAPER_URL_TODO`. The manuscript's LaTeX source is not distributed in this code repository.

## Citation

Author information and the public paper URL are pending. Please replace the placeholders before using this entry:

```bibtex
@misc{perceivesolvereason2027,
  title  = {Perceive, Solve, Reason: Progressive Capability Formation in Time Series-LLMs},
  author = {AUTHOR_NAMES_TODO},
  year   = {2027},
  url    = {PAPER_URL_TODO}
}
```

## Release checklist

- [x] Paper overview and architecture figure
- [x] Experiment settings and paper results
- [ ] Author names and public paper URL
- [x] Independently written reference training and evaluation code
- [ ] Original experiment code
- [ ] Dataset preparation instructions and checkpoints
- [x] Reference package, example configurations, and smoke tests
- [ ] Verified original dependencies and experiment commands

## License

The original reference code is released under the [MIT License](LICENSE). The paper-derived figure and reported result tables are not covered by the software license.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and result-reporting guidelines.
