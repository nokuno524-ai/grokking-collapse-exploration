#!/bin/bash
#SBATCH --job-name=esm2sae-explore-setup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/qzp4ta/grokking-collapse/logs/setup-%j.out
#SBATCH --error=/scratch/qzp4ta/grokking-collapse/logs/setup-%j.err

set -e

echo "=== Setting up Grokking-Collapse Experiment ==="

# Create project directory
mkdir -p /scratch/qzp4ta/grokking-collapse/{logs,results,src}

# Copy source files from local (assumes scp has been done)
# scp -r explorations/grokking-collapse/src/* rivanna:/scratch/qzp4ta/grokking-collapse/src/
# scp explorations/grokking-collapse/scripts/run_experiment.sh rivanna:/scratch/qzp4ta/grokking-collapse/
# scp explorations/grokking-collapse/pyproject.toml rivanna:/scratch/qzp4ta/grokking-collapse/

cd /scratch/qzp4ta/grokking-collapse

# Set up venv
uv venv .venv
source .venv/bin/activate

# Install dependencies (CPU-only torch is fine for this small experiment,
# but we want CUDA for GPU training)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy matplotlib

echo "=== Setup Complete ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
