#!/bin/bash
#SBATCH --job-name=grokking-multiseed
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-24
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/multiseed-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/multiseed-%A_%a.err

# 5 seeds × 5 conditions = 25 array tasks (indexed 0..24).
# See src/run_multi_seed.py::build_tasks for the (seed, condition) mapping.

export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

python3 -m src.run_multi_seed \
    --array-id "${SLURM_ARRAY_TASK_ID}" \
    --output-dir /scratch/qzp4ta/grokking-collapse/results/multi_seed
