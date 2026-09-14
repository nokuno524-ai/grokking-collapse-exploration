# Attention Visualization Tooling

This directory includes tools for analyzing attention patterns in checkpoints for modular arithmetic transformers.

## Usage

1. **Compare Attention:** Extracts attention patterns and computes stats across all checkpoints in a run.
   ```bash
   uv run python scripts/compare_attention.py --run-dir results/pure --out-dir analysis/attention/pure
   ```

2. **Cross-condition Comparison:** You can run `compare_attention.py` on different run directories and then aggregate the results. E.g.
   ```bash
   uv run python scripts/compare_attention.py --run-dir results/pure --out-dir analysis/attention/pure
   uv run python scripts/compare_attention.py --run-dir results/medium_collapse --out-dir analysis/attention/medium_collapse
   uv run python scripts/compare_attention.py --run-dir results/severe_collapse --out-dir analysis/attention/severe_collapse
   ```

3. **Aggregation:** A small CLI that walks a results directory, finds all runs, and produces one summary HTML report.
   ```bash
   uv run python scripts/aggregate_report.py --results-dir analysis/attention --out-file analysis/attention/report.html
   ```

## Development and Testing
You can run the tests using pytest:
```bash
uv run pytest tests/test_attention.py
```
