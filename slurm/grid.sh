#!/bin/bash
#SBATCH --job-name=grokking-grid
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-59
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/grid-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/grid-%A_%a.err

# 4 levels × 3 severities × 5 seeds = 60 tasks (indexed 0..59).
# See src/run_grid.py::build_tasks for the (level, severity, seed) mapping.

export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

python3 -m src.run_grid \
    --array-id "${SLURM_ARRAY_TASK_ID}" \
    --output-dir /scratch/qzp4ta/grokking-collapse/results/grid
