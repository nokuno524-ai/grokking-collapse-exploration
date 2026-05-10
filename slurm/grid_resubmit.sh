#!/bin/bash
#SBATCH --job-name=grokking-grid-fill
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=58,59
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/grid-fill-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/grid-fill-%A_%a.err

# Re-run the two missing grid tasks: array index 58 and 59
#   58 -> level=0.30, severity=0.9, seed=45
#   59 -> level=0.30, severity=0.9, seed=46
# These previously failed (12700932_58/59) with "No module named 'torch'"
# because .venv was empty at the time. Deps have been reinstalled.

export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

python3 -m src.run_grid \
    --array-id "${SLURM_ARRAY_TASK_ID}" \
    --output-dir /scratch/qzp4ta/grokking-collapse/results/grid
