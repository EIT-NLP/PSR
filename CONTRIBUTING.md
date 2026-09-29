# Contributing

Contributions to the reference implementation are welcome. Please keep a clear distinction between independently implemented code and the original paper experiments.

1. Describe which model path, data format, or evaluation task the change affects.
2. Add a small test for behavior that could regress, especially masks, labels, stage transitions, and metric parsing.
3. Run `python -m pytest -q` after installing `.[dev]`.
4. Avoid committing datasets, model weights, credentials, or outputs. The `.gitignore` covers common local artifacts.
5. For a claim that a result reproduces a manuscript table, include the dataset split, preprocessing, checkpoint, command, metric definition, and seed.

The repository's example JSONL files are only format illustrations. Dataset rights remain with their original providers.
