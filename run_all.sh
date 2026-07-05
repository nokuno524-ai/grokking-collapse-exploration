#!/bin/bash

echo "Starting automated experiment runner for all collapse levels..."

# Create necessary directories
mkdir -p results logs

# Array of condition names to run
CONDITIONS=("pure" "low_collapse" "medium_collapse" "high_collapse" "severe_collapse")

for cond in "${CONDITIONS[@]}"; do
    echo "========================================================"
    echo "Running condition: $cond"
    echo "========================================================"

    # Run the experiment via hydra overrides
    # Map the condition name to the proper variables (simplified for the script)
    case $cond in
      pure)
        ARGS="collapse_level=0.0 collapse_severity=0.5"
        ;;
      low_collapse)
        ARGS="collapse_level=0.05 collapse_severity=0.3"
        ;;
      medium_collapse)
        ARGS="collapse_level=0.15 collapse_severity=0.5"
        ;;
      high_collapse)
        ARGS="collapse_level=0.30 collapse_severity=0.7"
        ;;
      severe_collapse)
        ARGS="collapse_level=0.50 collapse_severity=0.9"
        ;;
    esac

    PYTHONPATH=. python src/train.py condition_name=$cond $ARGS > logs/${cond}.log 2>&1

    echo "Finished $cond. Log saved to logs/${cond}.log"
done

echo "Generating experiment dashboard..."
PYTHONPATH=. python src/analysis/generate_dashboard.py results dashboard.html

echo "All experiments completed successfully!"
