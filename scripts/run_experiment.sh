#!/bin/bash
#SBATCH --job-name=esm2sae-explore-grok
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/grok-%j.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/grok-%j.err

# Load CUDA
module load cuda/12.8.0

# Set up environment
cd /scratch/qzp4ta/grokking-collapse
source .venv/bin/activate

echo "=== Grokking-Collapse Experiment ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no gpu info')"
echo "Start: $(date)"

# Run all conditions sequentially
python src/train.py --all --max-steps 50000 --output-dir results

# Generate analysis
python src/progress_measures.py results/
python src/analysis.py results/

echo "=== Done: $(date) ==="
