# Analysis Suite for Grokking and Model Collapse

This directory describes the analysis tools added to explore the interplay between model collapse (synthetic data contamination) and grokking (delayed generalization).

## Overview of Findings

Based on the initial experiment logs:
- **Pure Data (0% Collapse):** The model successfully groks at step ~1400, achieving 100% test accuracy.
- **Low Collapse (5% Level, 0.3 Severity):** The model groks later at ~3100, achieving ~93% test accuracy.
- **Medium / High / Severe Collapse:** The models fail to grok completely.
- **Weight Norm Reduction:** A weight-norm reduction of 30–42% (peak vs final) correlates strongly with collapse severity (r = -0.82), confirming the theoretical predictions in `src/threshold_theory.py`.

## Regenerating Figures

You can regenerate the entire suite of figures and statistics by running the following Python scripts from the repository root. Ensure your environment has the necessary dependencies (`matplotlib`, `scipy`, `pandas`) by using the `uv` environment.

```bash
uv run python analysis/attention_evolution.py
uv run python analysis/weight_dynamics.py
uv run python analysis/phase_diagram.py
uv run python analysis/stats.py
```

### 1. Attention Evolution (`analysis/attention_evolution.py`)
Computes per-head attention entropy over a fixed probe batch across all training steps.
- Output: `analysis/attention/`
  - `entropy_curves_<condition>.png`: Line plots showing how each head's entropy changes, indicating specialization.
  - `entropy_heatmap_<condition>.png`: Heatmaps visualising entropy drops.

### 2. Weight Dynamics (`analysis/weight_dynamics.py`)
Traces the L2 norm of the model weights and the effective rank of embeddings and attention layers over training.
- Output: `analysis/weights/`
  - `norm_traj_<condition>.png`: L2 norm trajectories for various layers.
  - `rank_traj_<condition>.png`: Effective rank (SVD entropy) trajectories.
  - `norm_reduction_vs_severity.png`: Scatter plot mapping weight-norm reduction to collapse severity.

### 3. Phase Summary (`analysis/phase_diagram.py`)
Generates high-quality phase diagrams summarizing grokking outcomes across all collapse conditions and seeds.
- Output: `analysis/phase/`
  - `phase_diagram_grokking.[png|pdf]`: Collapse severity vs grokking step (failures plotted as 'x').
  - `phase_diagram_accuracy.[png|pdf]`: Norm reduction vs final accuracy scatter plot.

### 4. Statistics (`analysis/stats.py`)
Computes multi-seed aggregates (Mean ± SD) and runs non-parametric permutation tests to compare conditions (e.g., pure vs. low_collapse).
- Output: `analysis/stats/`
  - `summary_stats.csv`: Tabular statistics across conditions.
  - `statistical_tests.txt`: Results of permutation tests and Cohen's d effect sizes.
