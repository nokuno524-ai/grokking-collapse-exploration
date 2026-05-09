#!/bin/bash
#SBATCH --job-name=real-gen
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a100,gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/real-gen-%j.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/real-gen-%j.err

# Generate clean OpenWebText splits + every (ratio, seed) AI-contaminated
# mixture using GPT-2 XL as the generator. This is the heavy data prep step.

set -e
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /scratch/qzp4ta/grokking-collapse
source .venv_real/bin/activate

DATA_ROOT=/scratch/qzp4ta/grokking-collapse/data/contaminated_real

# 1. AI contamination (the headline contamination type)
python -m src.contamination_real.prepare_real_data \
    --ratios 0 5 15 30 50 80 \
    --seeds 0 1 2 \
    --gen-model gpt2-xl \
    --tokenizer gpt2-medium \
    --n-docs 200000 \
    --gen-batch-size 4 \
    --max-new-tokens 960 \
    --data-root "${DATA_ROOT}"

# 2. Random-token noise baseline (no GPU needed but sharing the job)
python -m src.contamination_real.prepare_real_data \
    --ratios 5 15 30 50 80 \
    --seeds 0 1 2 \
    --mode noise \
    --tokenizer gpt2-medium \
    --data-root "${DATA_ROOT}"

# 3. Scarcity baseline (just drops contaminated rows)
python -m src.contamination_real.prepare_real_data \
    --ratios 5 15 30 50 80 \
    --seeds 0 1 2 \
    --mode scarcity \
    --tokenizer gpt2-medium \
    --data-root "${DATA_ROOT}"

# 4. Self-contamination (same model family generates) — uses GPT-2 medium
python -m src.contamination_real.prepare_real_data \
    --ratios 5 15 30 50 80 \
    --seeds 0 1 2 \
    --mode self \
    --gen-model gpt2-medium \
    --tokenizer gpt2-medium \
    --data-root "${DATA_ROOT}"
