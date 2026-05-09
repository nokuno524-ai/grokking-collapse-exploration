#!/bin/bash
#SBATCH --job-name=grokking-baselines
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-49
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/baselines-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/baselines-%A_%a.err

# 25 noise tasks (indices 0..24) + 25 scarcity tasks (indices 25..49) = 50 total.
# Tasks 0..24 -> run_noise_baseline.py with --array-id (idx)
# Tasks 25..49 -> run_scarcity_baseline.py with --array-id (idx - 25)

export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

IDX="${SLURM_ARRAY_TASK_ID}"
if [ "${IDX}" -lt 25 ]; then
    SUB_IDX="${IDX}"
    echo "[baselines] noise task ${SUB_IDX}"
    python3 -m src.run_noise_baseline \
        --array-id "${SUB_IDX}" \
        --output-dir /scratch/qzp4ta/grokking-collapse/results/noise_baseline
else
    SUB_IDX=$((IDX - 25))
    echo "[baselines] scarcity task ${SUB_IDX}"
    python3 -m src.run_scarcity_baseline \
        --array-id "${SUB_IDX}" \
        --output-dir /scratch/qzp4ta/grokking-collapse/results/scarcity_baseline
fi
