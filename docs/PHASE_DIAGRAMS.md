# Weight Decay Phase Diagrams

This project includes a comprehensive phase diagram generator to explore the interplay between weight decay and synthetic data collapse.

## Methodology

We systematically vary weight decay and collapse severity:
- **Weight Decay (`wd`)**: {0.0, 0.001, 0.01, 0.1, 1.0}
- **Collapse Severity**: {pure, low_collapse, medium_collapse, severe_collapse, high_collapse}

For each configuration, a modular arithmetic model is trained. The core outcomes tracked are:
- **Grokking Outcome**: Whether the model successfully groks (reaches >= 95% test accuracy) before the max steps.
- **Peak Accuracy**: The highest test accuracy achieved during training.
- **Weight Norm Trajectory**: Tracked to analyze regularization effects.
- **Embedding Rank**: Used as a proxy for structural complexity.

## Running the Grid

To run the full grid of weight decay values across all collapse severities:
```bash
python src/run_wd_phase_diagram.py --max-steps 50000 --output-dir results/wd_phase_diagram
```

## Generating the Diagrams

Once the grid has finished running, generate the 2D phase diagrams:
```bash
python src/analysis/phase_diagram_wd.py --results-dir results/wd_phase_diagram --output-dir results/phase_diagrams
```

The script produces:
- `phase_diagram_grokking.png`: A heatmap illustrating the boundary between grokking (1) and non-grokking (0) regimes.
- `phase_diagram_peak_acc.png`: A continuous heatmap of peak validation accuracy.

## Preliminary Findings

- Higher weight decay values (e.g., `wd=1.0`) are crucial for grokking on pure data.
- However, as collapse severity increases (more corrupted labels/narrower distribution), the threshold for grokking shifts.
- With medium and severe collapse, grokking is universally prevented, regardless of the weight decay setting, forming a sharp boundary in the phase diagram.
