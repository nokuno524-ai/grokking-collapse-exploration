# Analysis Tools

This document outlines the usage of the new analysis tools added to this repository for exploring grokking under distributional collapse.

## Features

- Parse experiment logs (CSV or JSON lines format)
- Automatically catalog all completed experiments
- Detect the grokking point using threshold-based evaluation
- Plot training/validation loss and accuracy curves (with phase labeling)
- Compare multiple collapse levels simultaneously
- Generate a comprehensive HTML report containing metrics and plots

## Installation Requirements

Ensure you have the environment set up as detailed in the README. The specific analysis dependencies include:
- `pandas`
- `numpy`
- `matplotlib`
- `jsonlines`

## Modules

The main logic resides in `src/analysis/`:
- `parser.py`: Deals with file I/O, parsing logs, detecting grokking, and calculating metrics.
- `visualizer.py`: Houses all `matplotlib` logic to generate the experiment graphics and reports.

## Running the Automated Report

The easiest way to process your experiments is using the provided script:

```bash
python scripts/analyze_existing_results.py --results-dir results --output-dir results/analysis_report
```

This will:
1. Scan `results/` for all experiments.
2. Load all available JSON/CSV configurations and training histories.
3. Automatically generate an HTML report containing side-by-side graphs and performance details at `results/analysis_report/index.html`.

## Running the Tests

To run the full suite of testing to ensure parsing and visualization work correctly:

```bash
python -m pytest tests/
```
