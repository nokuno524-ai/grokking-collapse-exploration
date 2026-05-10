#!/bin/bash
#SBATCH --job-name=grok-trans
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/trans-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/trans-%A_%a.err
#SBATCH --array=0-9

# Surgical transplant rescue (Experiment A) across paired (pure, contam) seeds.
# Pairs are matched-seed: pure = wd1/noise0/seed_X, contam = wd1/noise0.15/seed_X.
# 5 seeds × 2 contam-noise levels (0.15 and 0.20) = 10 array tasks.

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

IDX="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS=(42 43 44 45 46)
NOISES=(0.15 0.2)
N_SEEDS=${#SEEDS[@]}
N_NOISES=${#NOISES[@]}

SEED_IDX=$(( IDX / N_NOISES ))
NOISE_IDX=$(( IDX % N_NOISES ))
SEED=${SEEDS[$SEED_IDX]}
NOISE=${NOISES[$NOISE_IDX]}

PURE_RUN="/scratch/qzp4ta/grokking-collapse/results/exp_c_grid/wd1/noise0/seed_${SEED}"
CONTAM_RUN="/scratch/qzp4ta/grokking-collapse/results/exp_c_grid/wd1/noise${NOISE}/seed_${SEED}"
OUT_DIR="/scratch/qzp4ta/grokking-collapse/analysis/transplant/wd1_n${NOISE}_s${SEED}"

echo "[trans] task=${IDX} seed=${SEED} noise=${NOISE}"
echo "[trans] pure=${PURE_RUN}"
echo "[trans] contam=${CONTAM_RUN}"

if [[ ! -d "$PURE_RUN" || ! -d "$CONTAM_RUN" ]]; then
    echo "[trans] ERROR: missing pure or contam run directory"
    exit 1
fi

python3 -m src.transplant_rescue \
    --pure-run "$PURE_RUN" \
    --contam-run "$CONTAM_RUN" \
    --output-dir "$OUT_DIR" \
    --rescue-steps 2000 \
    --seed "${SEED}"

echo "[trans] task=${IDX} done"
