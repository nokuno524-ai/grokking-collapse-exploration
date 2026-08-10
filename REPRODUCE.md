# Reproducibility Guide

This document provides step-by-step instructions to reproduce the theory tests, scaling law experiments, and compile the paper draft for this repository.

## 1. Setup Environment

Ensure you have `uv` installed, then set up the environment and install dependencies:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv pip install pytest
```

## 2. Validate Theoretical Models

We provide mathematical models representing the mutual information and weight norm trajectories governing grokking and collapse. To test them:

```bash
uv run pytest tests/test_theory.py
```

## 3. Run Extended Experiments

To run the new extended scaling laws (which test MLP, Transformer, and CNN architectures across different synthetic data ratios):

**Local Dry-Run:**
```bash
uv run python experiments/scaling_laws.py --model mlp --size small --ratio 0.5 --epochs 100
```

**Rivanna HPC (SLURM):**
```bash
sbatch slurm/run_scaling.sbatch
```

## 4. Compile Paper Draft

The theoretical and experimental findings are summarized in a LaTeX draft. To compile it:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
