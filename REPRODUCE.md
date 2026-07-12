# Experiment Reproduction Guide

This guide provides end-to-end instructions for reproducing the grokking-collapse experiments using the provided framework.

## 1. Environment Setup

The repository uses `uv` for fast python environment management. Ensure you are using python 3.10+.

```bash
# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install pyyaml torch numpy matplotlib scipy pandas seaborn pytest tabulate hydra-core plotly black isort flake8 mypy statsmodels jinja2 autopep8 pydantic scikit-learn tensorboard
```

## 2. Configuration Files

Experiment conditions are defined in YAML files inside the `configs/` directory.

The standard conditions provided are:
- `pure.yaml` (0% collapse)
- `low_collapse.yaml` (5% collapse, 0.3 severity)
- `medium_collapse.yaml` (15% collapse, 0.5 severity)
- `high_collapse.yaml` (30% collapse, 0.7 severity)
- `severe_collapse.yaml` (50% collapse, 0.9 severity)

These files dictate the model dimensions, data configurations, training epochs, weight decay, and the logging settings.

## 3. Running an Experiment

To run a single condition, use `runner.py`:

```bash
# Run the pure (baseline) condition
python runner.py --config configs/pure.yaml --output_dir results/pure_run

# Enable Weights & Biases logging
python runner.py --config configs/medium_collapse.yaml --output_dir results/medium_run --use_wandb
```

### Distributed Training

The runner is compatible with `torchrun` and `NCCL`. For example, to run on 4 GPUs:

```bash
torchrun --nproc_per_node=4 runner.py --config configs/pure.yaml --output_dir results/pure_distributed
```

## 4. Parameter Sweeps

To execute parameter sweeps (e.g. testing combinations of weight decay and noise fraction), you can parse the `configs/sweep.yaml` manually or create a short wrapper script to launch multiple runs by dynamically overwriting the arguments of the base YAML.

## 5. Result Aggregation

Once you have executed your runs (ideally over multiple random seeds for statistical validity), you can aggregate all results. This script recursively scans the `--results_dir` for `results.json` files, groups them by condition, and generates a `.csv` and a publication-ready `.tex` table.

```bash
python aggregate_results.py --results_dir results/ --output_file results/final_table.tex
```

## 6. Publication Plots

To generate publication-ready plots matching the paper's style (single/double column widths, LaTeX fonts if available, colorblind palettes):

```bash
python plots/paper_plots.py --results_dir results/ --output_dir plots/output/
```

This will produce `.png` and `.pdf` figures in `plots/output/`:
- `grokking_curves` (Train/Test accuracy over steps)
- `weight_norms` (L2 norm over steps)
- `loss_trajectories` (Train vs Test loss landscapes)
- `fourier_concentration` (Surrogate for embedding evolution)
