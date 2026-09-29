# Data and stage mapping

| Stage | Training sources | Paper-reported size | Evaluation |
| --- | --- | ---: | --- |
| 1 · Perceive | ChatTS temporal alignment subsets | ~105,000 | 525 manual questions and 1,616 synthetic questions |
| 2 · Solve | TSQA forecasting, imputation, anomaly detection, classification; TS Instruct classification | 158,159 total before 8:2 split | Four task-specific test sets |
| 3 · Reason | TSQA reasoning QA, ChatTS TSEvol + Instruct Follow, TSCognition, TS Instruct reasoning | See note below | QA evaluation and five-dimensional reasoning evaluation |

The paper lists Stage 3 components of 26,163 + 29,320 + 32,869 + 12,852 examples. These add up to **101,204**, while the main text reports **105,751**. The release needs a verified split manifest before a single total can be asserted.

The manuscript also reverses the labels “Evaluation Set A” and “Evaluation Set B” between the framework description and the RQ2 results tables. We therefore refer to them by task: **QA evaluation** (4,547 samples) and **five-dimensional evaluation** (6,163 samples), pending confirmation.

Datasets are not bundled here. Their licenses and access requirements belong to their respective providers. Source descriptions and counts are drawn from the paper's Dataset and Statistics Overview appendices.
