#!/bin/bash
#SBATCH --job-name=scaling_grokking
#SBATCH --output=logs/scaling_%j.out
#SBATCH --error=logs/scaling_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load environment
source .venv/bin/activate

# Run scaling grid
python scripts/run_scaling.py --out-file results_scaling/scaling_results.jsonl
