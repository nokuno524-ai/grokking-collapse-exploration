# Scaling Laws for Model Collapse and Grokking

## Hypothesis Space

The existing results demonstrate that model collapse prevents grokking in small LLMs on modular arithmetic tasks. The single most important external-validity question is how this effect scales. Does the critical threshold of collapse severity shift with model size or dataset size?

### Predictions

1. **Model Size Effect**:
   - *Hypothesis 1 (Capacity helps)*: Larger models have higher capacity to memorize noise but also to discover the underlying generative mechanism. They might grok faster or tolerate higher collapse severity before failing to grok.
   - *Hypothesis 2 (Capacity hurts)*: Larger models might overfit to the collapsed distribution more strongly (acting as "memorizing-only" models) and never transition to the grokking phase, making them *more* susceptible to collapse.

2. **Data Size Effect**:
   - More data generally speeds up grokking. However, in the presence of collapse, more *contaminated* data might reinforce the collapsed distribution.
   - *Prediction*: There is an interaction effect. For pure data, grokking onset will decrease with data size. For severely collapsed data, increasing data size might not rescue grokking if the signal-to-noise ratio is too low.

## Experimental Design

- **Grid Search**:
  - Model sizes: `tiny`, `small`, `base` (varying `d_model`, `n_heads`, `d_ff`).
  - Data fractions: `0.2`, `0.4`, `0.6`, `0.8`.
  - Collapse Severities: `pure`, `medium_collapse`, `severe_collapse`.

- **Metrics**:
  - *Grokking Onset*: The training step at which test accuracy first reaches 90%. If it never reaches 90% within the training budget, it is classified as "Failed".
  - Heatmaps will be generated plotting grokking onset against model size and data fraction, grouped by severity.

## Analysis Tooling
- `scripts/run_scaling.py`: Harness to run the grid and output step-wise loss/accuracy to JSONL.
- `scripts/plot_scaling.py`: Script to generate heatmaps from the JSONL output, using a 90% accuracy threshold for onset.
