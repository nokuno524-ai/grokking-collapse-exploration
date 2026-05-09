#!/bin/bash
#SBATCH --job-name=real-eval
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/real-eval-%j.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/real-eval-%j.err

# Run lm-eval-harness on every checkpoint produced by real_train.sbatch.
# Task list defaults to HellaSwag, ARC-Easy, PIQA, WinoGrande (zero-shot).

set -e
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /scratch/qzp4ta/grokking-collapse
source .venv_real/bin/activate

OUT=/scratch/qzp4ta/grokking-collapse/results/contamination_real

python -m src.contamination_real.eval_downstream \
    --output-dir "${OUT}" \
    --base-model gpt2-medium \
    --tasks hellaswag arc_easy piqa winogrande \
    --batch-size 8 \
    --final-only

# Aggregate everything into plots + summary.json
python -m src.contamination_real.analyze_real \
    --results-dir "${OUT}" \
    --toy-results-dir /scratch/qzp4ta/grokking-collapse/results/contamination
