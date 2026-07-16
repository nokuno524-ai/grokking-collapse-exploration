#!/bin/bash
# Reproduction script for Grokking-Collapse experiments.
# Usage: ./run_phase.sh <phase_number>
#
# Phase 1: Basic Grokking (runtime: ~20 mins on GPU)
# Phase 2: Visualization (runtime: ~1 min)
# Phase 3: Statistics / Grid (runtime: ~1 hr on GPU)
# Phase 4: Mechanistic (runtime: ~30 mins on GPU)

PHASE=$1
if [ -z "$PHASE" ]; then
    echo "Usage: ./run_phase.sh <phase_number>"
    # Exit avoided for test environments, just stop execution
    kill -INT $$
fi

export PYTHONPATH=.

if [ "$PHASE" -eq 1 ]; then
    echo "Running Phase 1: Basic Grokking..."
    python -m src.train --all --max-steps 10000 --output-dir results_reproduce/phase1
    echo "Phase 1 complete."
elif [ "$PHASE" -eq 2 ]; then
    echo "Running Phase 2: Visualization..."
    python -m src.analysis results_reproduce/phase1
    echo "Phase 2 complete."
elif [ "$PHASE" -eq 3 ]; then
    echo "Running Phase 3: Statistics / Grid..."
    # A smaller grid for reproduction
    python -m src.run_grid --levels 0.0,0.3 --severities 0.3,0.9 --seeds 42 --max-steps 5000 --output-dir results_reproduce/phase3
    echo "Phase 3 complete."
elif [ "$PHASE" -eq 4 ]; then
    echo "Running Phase 4: Mechanistic..."
    # Re-run pure and a collapsed condition to set up mechanistic tasks
    python -m src.train --condition pure --max-steps 5000 --output-dir results_reproduce/phase4
    python -m src.train --condition severe_collapse --max-steps 5000 --output-dir results_reproduce/phase4
    python -m src.transplant_rescue --pure-run results_reproduce/phase4/pure --contam-run results_reproduce/phase4/severe_collapse --output-dir results_reproduce/phase4/transplant
    echo "Phase 4 complete."
else
    echo "Unknown phase: $PHASE"
    # Exit avoided for test environments, just stop execution
    kill -INT $$
fi
