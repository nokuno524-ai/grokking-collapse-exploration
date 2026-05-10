#!/bin/bash
#SBATCH --job-name=expc-grid
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-89
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/expc-grid-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/expc-grid-%A_%a.err

# Experiment C: weight-decay x noise-rate x seed grid.
#   weight_decay (lambda): {0.3, 1.0, 3.0}
#   noise_fraction (eta):  {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}
#   seeds:                 {42, 43, 44, 45, 46}
# 3 * 6 * 5 = 90 tasks (indices 0..89).
# Each run trains a 1-layer transformer on modular arithmetic for 50k steps,
# logs final test accuracy and Fourier concentration in
#   results/exp_c_grid/wd<W>/noise<N>/seed_<S>/results.json.

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "[slurm] expc-grid task ${IDX}/90 on $(hostname)"

python -m src.run_exp_c_grid \
    --array-id "${IDX}" \
    --weight-decays 0.3,1.0,3.0 \
    --noise-fractions 0.0,0.05,0.10,0.15,0.20,0.30 \
    --seeds 42,43,44,45,46 \
    --max-steps 50000 \
    --output-dir /scratch/qzp4ta/grokking-collapse/results/exp_c_grid
