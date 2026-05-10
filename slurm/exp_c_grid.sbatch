#!/bin/bash
#SBATCH --job-name=grok-expc
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/expc-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/expc-%A_%a.err
#SBATCH --array=0-89%30

# Experiment C: weight-decay x noise-rate grid (toy modular arithmetic).
# Default grid: wd in {0.3, 1.0, 3.0} x noise in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30} x 5 seeds = 90 tasks.
# Re-run with --array=0-N if you change the grid.

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "[expc] task=${IDX} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"

python3 -m src.run_exp_c_grid \
    --array-id "${IDX}" \
    --output-dir /scratch/qzp4ta/grokking-collapse/results/exp_c_grid \
    --max-steps 50000

echo "[expc] task=${IDX} done"
