#!/bin/bash
#SBATCH --job-name=phase-diag
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/phasediag-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/phasediag-%A_%a.err
#SBATCH --array=0-479%50

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "[phasediag] task=${IDX} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"

python3 -m src.sweep.driver \
    --array-id "${IDX}" \
    --output-dir results/phase_diagram \
    --max-steps 50000

echo "[phasediag] task=${IDX} done"
