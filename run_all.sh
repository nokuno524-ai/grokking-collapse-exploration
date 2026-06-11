#!/bin/bash
set -e

echo "=================================================="
echo "Grokking & Collapse Scaling Experiments"
echo "=================================================="

# 1. Setup environment
echo "[1/4] Setting up environment..."
uv venv .venv || python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt || pip install -r requirements.txt

# 2. Run Tests
echo "[2/4] Running tests to verify metric correctness..."
PYTHONPATH=. pytest tests/

# 3. Run Scaling Experiments
echo "[3/4] Running scaling experiments..."
# Create a runner script to execute the scaling loop
cat << 'EOF' > run_scaling.py
import sys
import os
sys.path.append(os.path.abspath('.'))
from experiments.scaling import run_scaling_experiments, plot_scaling_laws, ScalingExperimentConfig

config = ScalingExperimentConfig(max_steps=10000, eval_every=500)
results = run_scaling_experiments(config)
plot_scaling_laws(results, "results/scaling/plots")
EOF

python run_scaling.py

# 4. Success message
echo "[4/4] Done! Results saved to results/scaling/"
echo "=================================================="
echo "Check EXPERIMENTS.md for details on analyzing results."
