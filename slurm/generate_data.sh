#!/bin/bash
#SBATCH --job-name=contam-gen
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40,gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/contam-gen-%j.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/contam-gen-%j.err

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

# Generate the clean splits + every (ratio, seed) contaminated mixture
python -m src.contamination.prepare_data \
    --ratios 0 10 30 50 80 100 \
    --seeds 0 1 2 \
    --data-root /scratch/qzp4ta/grokking-collapse/data/contaminated
