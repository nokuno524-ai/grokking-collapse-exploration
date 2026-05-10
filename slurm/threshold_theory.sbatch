#!/bin/bash
#SBATCH --job-name=grok-theory
#SBATCH --account=zhangmlgroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/theory-%j.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/theory-%j.err

# Threshold theory analysis: empirical fit of η*(λ) on results/exp_c_grid/.
# Pure analysis pass — no GPU needed.

set -e
export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

python3 -m src.threshold_theory \
    --grid-dir /scratch/qzp4ta/grokking-collapse/results/exp_c_grid \
    --output-dir /scratch/qzp4ta/grokking-collapse/analysis

python3 -m src.leading_indicator_test \
    --grid-dir /scratch/qzp4ta/grokking-collapse/results/exp_c_grid \
    --output-dir /scratch/qzp4ta/grokking-collapse/analysis

echo "[theory] done"
