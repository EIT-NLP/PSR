# Results

Values below are transcribed from the manuscript, not rerun from released code. Machine-readable versions are in [`../results/`](../results/).

## Out-of-domain MTBench

| Model | Weather MCQA ↑ | Weather prediction ↑ | Finance 7d 3-way ↑ | Finance 7d 5-way ↑ | Finance 7d MSE ↓ | Finance 30d 3-way ↑ | Finance 30d 5-way ↑ | Finance 30d MSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TimeOmni-1-7B | 41.67 | 51.40 | 40.76 | 34.97 | 0.622 | 37.39 | 20.00 | 1.490 |
| ITFormer-7B | 57.14 | 36.26 | 40.14 | 33.57 | 0.174 | 51.03 | 31.72 | 0.907 |
| ChatTS-7B | 56.21 | 28.79 | 19.71 | 16.09 | 2.047 | 19.38 | 13.46 | 3.342 |
| PATRA-7B | 60.46 | 52.04 | 63.33 | **60.80** | 0.191 | 54.39 | 43.70 | 0.822 |
| **TS-LLM (Pre-align)** | **67.24** | **57.17** | **68.19** | 57.82 | 0.187 | 60.29 | 41.51 | 0.837 |
| **TS-LLM (Post-align)** | 65.92 | 52.76 | 59.88 | 55.43 | **0.160** | **65.78** | **47.16** | **0.744** |

Source: paper's out-of-domain results table (`ood_results`). The paper does not specify units for the accuracy-like MTBench columns in this table; values are copied as printed.

## Perception generalization

| Model | TSE anomaly | TSE causality | TSE noise | TSE pattern | TSE similarity | CaTS caption retrieval | CaTS TS retrieval | CaTS amplitude | CaTS mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Paper | 28.0 | 52.0 | 26.0 | 47.0 | 45.0 | 44.0 | 29.0 | 43.0 | 35.0 |
| TimeOmni-1 | 42.6 | 37.5 | 56.0 | 56.4 | 46.7 | 57.0 | 37.0 | 92.5 | 67.5 |
| **TS-LLM (Pre-align)** | **49.1** | 47.2 | **60.7** | **61.6** | **55.0** | 52.0 | **46.0** | 82.5 | **77.5** |

Source: paper's perception generalization table (`perception_generalization`).

## Progressive training

Across the controlled experiments in the paper's results section, Stage 1 → 2 improves most Stage 2 metrics compared with Stage 2 alone. For Pre-align, forecasting MSE improves from 1.368 to 1.328 and classification accuracy from 0.567 to 0.628. The full Stage 1 → 2 → 3 path produces the strongest overall high-level reasoning results in both alignment settings, especially at smaller Stage 3 data scales. For example, Pre-align five-dimensional accuracy reaches 82.5 / 79.1 / 70.7 / 65.4 / 57.2 on decoding / grounding / inferring / extrapolating / acting. The [Stage 2](../results/task_solving_stage2.csv), [five-dimensional reasoning](../results/reasoning_five_dimensions.csv), and [QA reasoning](../results/reasoning_qa.csv) tables contain all values from the corresponding main-paper comparisons. Additional ablations remain in the manuscript.
