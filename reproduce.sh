#!/bin/bash
set -e

echo "=============================================="
echo "Grokking & Collapse: End-to-End Pipeline"
echo "=============================================="

# This script assumes uv is installed and environments are setup
# Usage: ./reproduce.sh [--quick]

MAX_STEPS=50000
if [ "$1" == "--quick" ]; then
    MAX_STEPS=500
    echo "Running in quick mode with MAX_STEPS=$MAX_STEPS"
fi

echo -e "\n1. Running base model training (Pure Data)"
uv run python src/train.py --condition pure --max-steps $MAX_STEPS

echo -e "\n2. Running Mechanistic Analysis (Circuits, Rank, Logit Lens)"
# Pass the run directory
uv run python src/analysis/circuits.py --run-dir results/pure
uv run python src/analysis/weights.py --run-dir results/pure
uv run python src/analysis/logit_lens.py --run-dir results/pure

echo -e "\n3. Running Interpolation Study (Threshold detection)"
uv run python src/interpolation_study.py --max-steps $MAX_STEPS

echo -e "\n4. Running Recovery Interventions"
uv run python src/experiments/recovery.py --max-steps $MAX_STEPS

echo -e "\n5. Running Curriculum Learning"
uv run python src/experiments/curriculum.py --max-steps $MAX_STEPS

echo -e "\n6. Generating Publication Figures"
uv run python scripts/make_paper_figures.py

echo -e "\nPipeline complete! Results are in RESULTS.md and figures/."
