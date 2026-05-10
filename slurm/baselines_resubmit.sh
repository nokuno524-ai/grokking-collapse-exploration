#!/bin/bash
#SBATCH --job-name=grokking-baselines-fill
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=3-49
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/baselines-fill-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/baselines-fill-%A_%a.err

# Re-run baseline tasks 3..49.
#   Indices 0..24 -> noise baseline (5 fractions × 5 seeds)
#   Indices 25..49 -> scarcity baseline (5 fractions × 5 seeds)
# Tasks 0,1,2 (noise=0, seeds 42/43/44) already completed under 12700933.
# The remaining 47 tasks failed previously (12700933) with "No module named 'torch'"
# because .venv was empty. Deps have been reinstalled.

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
