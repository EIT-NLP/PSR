# Reproducibility

These settings are transcribed from the manuscript's reproducibility appendix. The repository's reference code is independently written; exact versions of the original software, original preprocessing scripts, checkpoints, and original executable commands are unavailable.

For a starting software environment, create the Conda environment, install the PyTorch build appropriate for your GPU and CUDA driver using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install the other reported framework packages:

```bash
conda env create -f environment.example.yml
conda activate perceive-solve-reason
# Install the appropriate PyTorch build here, following the official installer.
python -m pip install -e .
```

The example environment and package constraints are unverified against the original experiments. DeepSpeed is optional for this reference package and can be configured through Accelerate for larger runs.

| Setting | Reported value |
| --- | --- |
| Primary LLM | Qwen3-4B |
| Supplementary LLM | Qwen3-1.7B |
| Hardware | 2 × NVIDIA A800 GPUs |
| Stack | PyTorch, Hugging Face Transformers, DeepSpeed, Accelerate |
| Precision | bfloat16 |
| Optimizer | AdamW |
| Learning rate | 2 × 10⁻⁵ |
| Weight decay | 0.01 |
| Batch size / GPU | 4 |
| Gradient accumulation | 4 steps |
| Effective batch size | 32 |
| Gradient clipping | 1.0 |
| Scheduler | Cosine decay |
| Warmup ratio | 0.0 |
| Random seed | 2026 |

All three stages use supervised fine-tuning under the same optimization configuration unless otherwise specified in the paper.

**Pre-align:** Stage 1 updates the temporal encoder and alignment module. The encoder is frozen in Stages 2 and 3 while the remaining alignment components are updated. **Post-align:** the encoder and LLM are jointly optimized. **Textualization:** time series values are passed as text and only the LLM is fine-tuned.

The paper uses Stage 1 categorical F1 and numerical relative accuracy; Stage 2 forecasting/imputation MSE and anomaly detection/classification accuracy; and Stage 3 multiple-choice accuracy, plus accuracy, ROUGE-L, and BLEU for the QA evaluation. The MTBench experiments are out-of-domain evaluation only.

## Before a runnable release

The following details still need to be supplied by the original experiment code or authors: exact dependency versions, source repository provenance, checkpoint identifiers, dataset download and preprocessing commands, split manifests, original prompt templates, full model configuration, and launch commands. The reference package demonstrates the method but does not establish a faithful numerical reproduction.
