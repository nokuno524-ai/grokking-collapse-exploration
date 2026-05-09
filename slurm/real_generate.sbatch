#!/bin/bash
#SBATCH --job-name=real-gen
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a100,gpu-a40,gpu-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/real-gen-%j.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/real-gen-%j.err

# Headline AI-contamination data prep:
#   200K OpenWebText docs, 6 ratios x 3 seeds, GPT-2 XL generator.
# Auxiliary baselines (noise / scarcity / self) are produced by submitting
# this script with MODE=noise|scarcity|self, e.g.:
#   sbatch --export=ALL,MODE=noise slurm/real_generate.sbatch
#   sbatch --export=ALL,MODE=scarcity slurm/real_generate.sbatch
#   sbatch --export=ALL,MODE=self,GEN_MODEL=gpt2-medium slurm/real_generate.sbatch

set -e
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1
cd /scratch/qzp4ta/grokking-collapse
source .venv_real/bin/activate

DATA_ROOT=/scratch/qzp4ta/grokking-collapse/data/contaminated_real
MODE=${MODE:-ai}
GEN_MODEL=${GEN_MODEL:-gpt2-xl}

echo "[slurm] MODE=${MODE} GEN_MODEL=${GEN_MODEL}"
nvidia-smi || true

python -m src.contamination_real.prepare_real_data \
    --ratios 0 5 15 30 50 80 \
    --seeds 0 1 2 \
    --mode "${MODE}" \
    --gen-model "${GEN_MODEL}" \
    --tokenizer gpt2-medium \
    --n-docs 200000 \
    --gen-batch-size 8 \
    --max-new-tokens 960 \
    --max-seq-len 1024 \
    --data-root "${DATA_ROOT}"
