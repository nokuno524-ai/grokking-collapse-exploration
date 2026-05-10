#!/bin/bash
#SBATCH --job-name=contam-gen4
#SBATCH --account=zhangmlgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/contam-gen4-%j.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/contam-gen4-%j.err

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source /scratch/qzp4ta/grokking-collapse/.venv/bin/activate

# Generate only the missing ratios (50, 80, 100) — 0, 10, 30 already done
python -m src.contamination.prepare_data \
    --ratios 50 80 100 \
    --seeds 0 1 2 \
    --data-root /scratch/qzp4ta/grokking-collapse/data/contaminated
