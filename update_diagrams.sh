#!/bin/bash
# Re-runs analysis scripts idempotently over the results directory to generate phase diagrams and summaries.
set -e

# Base parameters
RESULTS_DIR="${1:-results/grid}"
OUT_DIR="${2:-analysis/scaling}"

echo "Updating Phase Diagrams & Scaling Fits..."
PYTHONPATH=. uv run --with numpy --with scipy --with matplotlib --with pandas --no-project python src/analysis/scaling_boundary.py \
    --results-dir "$RESULTS_DIR" \
    --out-dir "$OUT_DIR"

echo "Done. Outputs generated in $OUT_DIR."
