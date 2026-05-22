# Reproducibility Guide

This directory contains the scripts and locked dependencies necessary to reproduce the main experiments and figures for the "Grokking Cliffs" paper.

## Requirements

- Python 3.10+
- `uv` package manager (recommended for fast dependency resolution)

## Running Experiments

To run all experiments from scratch, use the provided `run_all.sh` script. Note that actual training runs can be computationally intensive; if you are running on a cluster, you may want to submit the individual SLURM array jobs instead.

```bash
# Set up environment, install exact dependency versions, and run scripts sequentially
bash reproduce/run_all.sh
```

### Expected Runtimes

- **Toy Multi-seed Grid (90 runs)**: ~3-5 minutes per run on a single CPU core. Sequentially: ~4-7 hours total.
- **Hyperparameter Sweeps**: ~1-2 hours depending on grid density.
- **Figure Generation**: ~5 minutes.

## Outputs

After execution, all generated data will be in the `results/` directory, and the generated figures and CSV summaries will be saved in `analysis/`.
