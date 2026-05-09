#!/bin/bash
#SBATCH --job-name=real-train
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000,gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/real-train-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/real-train-%A_%a.err
#SBATCH --array=0-17

# 6 ratios × 3 seeds = 18 tasks. Index = ratio_idx * 3 + seed_idx.
# Pass MODE=noise|scarcity|self|external as an env var to run a baseline
# instead of the headline AI-contamination sweep.

RATIOS=(0 5 15 30 50 80)
SEEDS=(0 1 2)
MODE=${MODE:-ai}

set -e
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /scratch/qzp4ta/grokking-collapse
source .venv_real/bin/activate

IDX=${SLURM_ARRAY_TASK_ID:-0}
R_IDX=$(( IDX / 3 ))
S_IDX=$(( IDX % 3 ))
RATIO=${RATIOS[$R_IDX]}
SEED=${SEEDS[$S_IDX]}

OUT=/scratch/qzp4ta/grokking-collapse/results/contamination_real
DATA=/scratch/qzp4ta/grokking-collapse/data/contaminated_real

echo "[slurm] task=$IDX ratio=${RATIO}% seed=$SEED mode=${MODE}"
nvidia-smi || true

python -m src.contamination_real.train_real \
    --ratio "$RATIO" --seed "$SEED" --mode "$MODE" \
    --base-model gpt2-medium \
    --data-root "${DATA}" \
    --output-dir "${OUT}" \
    --max-steps 50000 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 5e-5 \
    --warmup-steps 1000 \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --log-every 500 \
    --ckpt-every 10000 \
    --lora-r 16 \
    --lora-alpha 32
