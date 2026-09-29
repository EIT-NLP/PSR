# Data and stage mapping

| Stage | Training sources | Paper-reported size | Evaluation |
| --- | --- | ---: | --- |
| 1 · Perceive | ChatTS temporal alignment subsets | ~105,000 | 525 manual questions and 1,616 synthetic questions |
| 2 · Solve | TSQA forecasting, imputation, anomaly detection, classification; TS Instruct classification | 158,159 total before 8:2 split | Four task-specific test sets |
| 3 · Reason | TSQA reasoning QA, ChatTS TSEvol + Instruct Follow, TSCognition, TS Instruct reasoning | See note below | QA evaluation and five-dimensional reasoning evaluation |

The paper lists Stage 3 components of 26,163 + 29,320 + 32,869 + 12,852 examples. These add up to **101,204**, while the main text reports **105,751**. The release needs a verified split manifest before a single total can be asserted.

The manuscript also reverses the labels “Evaluation Set A” and “Evaluation Set B” between the framework description and the RQ2 results tables. We therefore refer to them by task: **QA evaluation** (4,547 samples) and **five-dimensional evaluation** (6,163 samples), pending confirmation.

Datasets are not bundled here. Their licenses and access requirements belong to their respective providers. Source descriptions and counts are drawn from the paper's Dataset and Statistics Overview appendices.

## Primary dataset and benchmark links

| Resource | Role in the paper | Link type and source |
| --- | --- | --- |
| ChatTS | Stage 1 alignment data; Stage 3 TSEvol / Instruct Follow | [Training dataset](https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset), linked by the [ChatTS project](https://github.com/NetManAIOps/ChatTS) |
| TSQA (Time-MQA) | Stage 2 numerical tasks; Stage 3 QA | [Time-MQA release hub](https://huggingface.co/Time-MQA), identified in the [paper](https://aclanthology.org/2025.acl-long.1437/) |
| TS Instruct (Chat-TS) | Stage 2 classification; Stage 3 reasoning | [TS Instruct dataset](https://huggingface.co/datasets/PaulQ1/TS_Instruct_Reasoning_V1), listed in the [paper's Assets appendix](https://aclanthology.org/2026.eacl-long.263/) |
| TSCognition | Stage 3 five reasoning dimensions | [Paper](https://arxiv.org/abs/2606.22126); a working public dataset download page has not been confirmed |
| TimeSeriesExam | External perception benchmark | [Dataset](https://huggingface.co/datasets/AutonLab/TimeSeriesExam), identified by the [paper](https://arxiv.org/abs/2410.14752) |
| CaTS-Bench | External perception benchmark | [Dataset](https://huggingface.co/datasets/mhfisher/CaTSBench), linked by the [project](https://github.com/LuckerZOfficiaL/CaTS-Bench) |
| MTBench | Out-of-domain weather and finance evaluation | [Paper](https://arxiv.org/abs/2503.16858); a working public dataset download page has not been confirmed |

The Stage 1 manual/synthetic questions and the paper's exact training/evaluation split manifests do not have verified standalone download links. The links above do not establish that a provider's current release matches the precise paper split.

## Public source datasets underlying TSCognition

TSCognition draws time series from 15 public datasets. These are **upstream raw-data sources**, not a substitute for the paper's generated TSCognition QA split:

| Domain | Sources |
| --- | --- |
| Energy | [Buildings-900K](https://data.openei.org/submissions/5859), [UMass Smart*](https://traces.cs.umass.edu/docs/traces/smartstar/) |
| Retail | [Favorita Sales](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data), [Dominick](https://www.chicagobooth.edu/research/kilts/datasets/dominicks) |
| Healthcare | [Project Tycho](https://www.tycho.pitt.edu/), [JHU CSSE COVID-19](https://github.com/CSSEGISandData/COVID-19) |
| Transportation | [LargeST](https://github.com/liuxu77/LargeST), [METR-LA](https://github.com/liyaguang/DCRNN) |
| Climate | [SubseasonalClimateUSA](https://github.com/microsoft/subseasonal_data), [Oikolab Weather](https://docs.oikolab.com/datasets/) |
| Finance | [Bitcoin](https://zenodo.org/records/5122101) |
| Web | [Web Traffic](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/data) |
| Industry | [WADI](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/wadi/), [C-MAPSS](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data), [Paderborn](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter) |
