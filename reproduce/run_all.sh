#!/bin/bash
set -e

echo "=== Grokking Collapse Reproducibility Pack ==="

# 1. Environment Setup
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi
source .venv/bin/activate
echo "Installing dependencies..."
uv pip install -r reproduce/requirements.txt

# 2. Main Experiments
echo "Running multi-seed experiments..."
python src/run_multi_seed.py

echo "Running hyperparameter sensitivity sweeps..."
python analysis/hyperparam_sensitivity.py --generate-configs
# Note: Actually running these sequentially takes hours to days on CPU.
# This script executes them as intended for reproducibility.
python analysis/hyperparam_sensitivity.py --run-configs

echo "Running seed analysis baseline..."
python analysis/seed_analysis.py --generate-configs
python analysis/seed_analysis.py --run-configs

# 3. Figure Generation
echo "Generating analysis figures..."
python src/analysis.py
python analysis/seed_analysis.py --plot
python analysis/hyperparam_sensitivity.py --plot

echo "=== Reproduction Complete ==="
