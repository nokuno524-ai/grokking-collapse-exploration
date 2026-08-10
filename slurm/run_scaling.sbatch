#!/bin/bash
#SBATCH --job-name=scaling_laws
#SBATCH --output=logs/scaling_%A_%a.out
#SBATCH --error=logs/scaling_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the virtual environment
# We assume standard setup procedure has run. In the repo run_in_venv.sh assumes .venv_real
# but the standard uv venv makes .venv. Let's rely on standard activation.
if [ -d ".venv_real" ]; then
    source .venv_real/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found."
    exit 1
fi

# Example sweep over model sizes
SIZES=("small" "medium" "large")
MODELS=("transformer" "mlp")
RATIOS=(0.0 0.5 1.0)

# Simplistic approach to demonstrate sbatch readiness
MODEL=${MODELS[0]}
SIZE=${SIZES[0]}
RATIO=${RATIOS[0]}

echo "Running scaling experiment for model=$MODEL size=$SIZE ratio=$RATIO"

python experiments/scaling_laws.py --model $MODEL --size $SIZE --ratio $RATIO --epochs 50000
