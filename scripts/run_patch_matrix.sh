#!/bin/bash
# Runs the full patch matrix for pure vs medium_collapse across seeds 42-46.

export PYTHONPATH=.

OUTPUT_DIR="analysis/patch_matrix"
mkdir -p "$OUTPUT_DIR"

SEEDS=(42 43 44 45 46)

for SEED in "${SEEDS[@]}"; do
    PURE_RUN="results/multi_seed/${SEED}/pure"
    CONTAM_RUN="results/multi_seed/${SEED}/medium_collapse"

    # Pure -> Collapse (Rescue)
    echo "Running Patch Matrix for seed ${SEED}"
    python src/transplant/head_transplant.py \
        --donor-runs "$PURE_RUN" \
        --recipient-runs "$CONTAM_RUN" \
        --output-dir "$OUTPUT_DIR/seed${SEED}" \
        --rescue-steps 0
done
