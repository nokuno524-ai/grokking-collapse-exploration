# Grokking-Collapse Reproduction Guide

This repository includes a full reproduction package to verify the findings on model collapse vs grokking.

## Quick Start

We provide bash scripts to automate the reproduction pipeline.

```bash
# Set up the environment
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt # Or install the packages directly

# Run all phases sequentially (estimated runtime: ~2 hours on a single GPU)
./reproduce/run_all.sh
```

## Phases

You can also run individual phases using the `run_phase.sh` script:

1. **Phase 1: Basic Grokking** (`./reproduce/run_phase.sh 1`)
   - Re-runs the pure baseline and simple collapse conditions.
   - Outputs: `results_reproduce/phase1`

2. **Phase 2: Visualization** (`./reproduce/run_phase.sh 2`)
   - Generates trajectory plots and comparison bar charts.
   - Outputs: `results_reproduce/phase1/*.png`

3. **Phase 3: Statistics / Grid** (`./reproduce/run_phase.sh 3`)
   - Runs a scaled-down grid of `collapse_level` x `collapse_severity` to confirm monotonic scaling properties.
   - Outputs: `results_reproduce/phase3`

4. **Phase 4: Mechanistic** (`./reproduce/run_phase.sh 4`)
   - Performs a surgical circuit transplant between a pure and severely collapsed network to identify where the collapse inhibits grokking (e.g. at the representation level or optimization level).
   - Outputs: `results_reproduce/phase4`

## Jupyter Notebooks

For interactive exploration, we provide Colab-ready notebooks in `notebooks/`:
- `01_basic_grokking.ipynb`: Shows the grokking delay in standard models.
- `02_collapse_effect.ipynb`: Shows how collapse delays/prevents grokking.
- `03_scaling_analysis.ipynb`: Visualizes the grokking cliff phase transitions.

## Packaging Results

Once you have run the experiments, use `package_results.py` to bundle the metrics and plots into a distributable archive while skipping heavy checkpoint files:

```bash
python package_results.py --results-dir results_reproduce --output-tar grokking_collapse_results.tar.gz
```
