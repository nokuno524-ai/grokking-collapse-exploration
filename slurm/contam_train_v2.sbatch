#!/bin/bash
#SBATCH --job-name=contam-train
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/contam-train-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/contam-train-%A_%a.err
#SBATCH --array=0-8

# Array layout: 3 ratios (0,10,30) x 3 seeds = 9 tasks
RATIOS=(0 10 30)
SEEDS=(0 1 2)

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

IDX=${SLURM_ARRAY_TASK_ID:-0}
R_IDX=$(( IDX / 3 ))
S_IDX=$(( IDX % 3 ))
RATIO=${RATIOS[$R_IDX]}
SEED=${SEEDS[$S_IDX]}

echo "[slurm] task=$IDX ratio=${RATIO}% seed=$SEED"

python -m src.contamination.train_contaminated \
    --ratio "$RATIO" \
    --seed "$SEED" \
    --output-dir /scratch/qzp4ta/grokking-collapse/results/contamination \
    --max-steps 50000 \
    --batch-size 8 \
    --lr 5e-5 \
    --warmup-steps 1000
