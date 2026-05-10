#!/bin/bash
#SBATCH --job-name=grok-realgen
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a100,gpu-a40,gpu-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/realgen-%A_%a.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/realgen-%A_%a.err
#SBATCH --array=0-17%4

# Real-LM data generation v2: SPLIT one (ratio, seed) per array task.
# Original v1 (real_generate.sbatch) tried to do all 18 cells in one 12 h job and
# was cancelled at time limit. v2 splits the work so each task fits in 8 h, and
# prepare_real_data.py already short-circuits if the output dir exists, so a
# crashed task can be re-submitted with --array=0-17 and only do the missing
# cells.
#
# Grid: ratios {0, 5, 15, 30, 50, 80} × seeds {0, 1, 2} = 18 tasks.

set -e
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
cd /scratch/qzp4ta/grokking-collapse
source .venv_real/bin/activate

DATA_ROOT=/scratch/qzp4ta/grokking-collapse/data/contaminated_real
MODE=${MODE:-ai}
GEN_MODEL=${GEN_MODEL:-gpt2-xl}

RATIOS=(0 5 15 30 50 80)
SEEDS=(0 1 2)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
N_SEEDS=${#SEEDS[@]}
RATIO_IDX=$(( IDX / N_SEEDS ))
SEED_IDX=$(( IDX % N_SEEDS ))
RATIO=${RATIOS[$RATIO_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[realgen-v2] task=${IDX}  ratio=${RATIO}  seed=${SEED}  mode=${MODE}  gen=${GEN_MODEL}"
nvidia-smi || true

OUT_DIR="${DATA_ROOT}/ratio_${RATIO}/seed_${SEED}"
if [[ "${MODE}" != "ai" ]]; then
    OUT_DIR="${DATA_ROOT}/mode_${MODE}/ratio_${RATIO}/seed_${SEED}"
fi
if [[ -d "${OUT_DIR}" ]] && [[ -n "$(ls -A "${OUT_DIR}" 2>/dev/null)" ]]; then
    echo "[realgen-v2] ${OUT_DIR} already populated; skipping."
    exit 0
fi

python -m src.contamination_real.prepare_real_data \
    --ratios "${RATIO}" \
    --seeds "${SEED}" \
    --mode "${MODE}" \
    --gen-model "${GEN_MODEL}" \
    --tokenizer gpt2-medium \
    --n-docs 200000 \
    --gen-batch-size 8 \
    --max-new-tokens 960 \
    --max-seq-len 1024 \
    --data-root "${DATA_ROOT}"

echo "[realgen-v2] task=${IDX} done"
