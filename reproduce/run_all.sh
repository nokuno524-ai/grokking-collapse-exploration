#!/bin/bash
# Reproduction wrapper script for Grokking-Collapse experiments.
# Runs all phases sequentially.
# Total Estimated Runtime: ~2 hours on a single GPU (e.g., A40/A6000).
# Resource Requirements: 1 GPU, 16GB RAM, 10GB disk space.

echo "Starting full reproduction pipeline..."

bash reproduce/run_phase.sh 1
bash reproduce/run_phase.sh 2
bash reproduce/run_phase.sh 3
bash reproduce/run_phase.sh 4

echo "Full reproduction pipeline complete!"
