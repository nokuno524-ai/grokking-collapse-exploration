#!/bin/bash
#SBATCH --job-name=grokking-seeds
#SBATCH --account=zhangmlgroup
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/seeds-%j.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/seeds-%j.err

export PYTHONUNBUFFERED=1
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

# Run all conditions (pure + all collapse levels) with 50K steps
python3 -m src.train --all --max-steps 50000 --output-dir /scratch/qzp4ta/grokking-collapse/results/seed_sweep
