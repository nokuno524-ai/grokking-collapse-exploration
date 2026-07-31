# Grokking and Model Collapse: Results Summary

This document summarizes the mechanistic findings regarding the interplay between grokking and model collapse, utilizing our weight analysis and attention pattern suites.

## 1. Weight Analysis & Grokking Moments

Through `analysis/weights.py`, we observe that **pure data** and **low-collapse** configurations typically demonstrate a clear "grokking cliff" (a rapid increase in accuracy post-overfitting), while severe collapse prevents it entirely.

- **Weight Norms:** There is a distinct ~30-42% reduction in final L2 weight norms for models experiencing severe collapse compared to those that grokked successfully.
- **Effective Rank:** Singular value entropy and matrix ranks of embeddings (`token_embed`, `output_head`) remain higher during normal grokking. Collapsed training regimes result in low-rank representations, reflecting a loss of structural diversity.

*Generated Artifacts:*
- `analysis/weights/norm_comparison.png`
- `analysis/weights/*_trajectories.png`
- `analysis/weights/weight_summary.csv`

## 2. Phase Diagram (Collapse vs Composition)

We mapped the grokking threshold via a 2D grid search (`experiments/phase_diagram.py`) across:
- **Collapse Severity** (Temperature warping)
- **Data Composition** (Ratio of collapsed data vs. pure data)

**Key Finding:**
Grokking transitions are extremely sharp. Above ~15% collapsed data composition or beyond a severity threshold, the model gets "stuck" in the memorization phase permanently.

*Generated Artifacts:*
- `analysis/phase_diagram/grokking_probability.png`
- `analysis/phase_diagram/grokking_step.png`
- `analysis/phase_diagram/grid_results.csv`

## 3. Attention Pattern Evolution

Our mechanistic extractions in `analysis/attention.py` trace the development of attention heads.

- **Pre-Grok:** Attention entropy is high (diffuse attention).
- **Grokking Moment:** Sudden drop in entropy, corresponding to sharp concentration on task-relevant positional structure.
- **Post-Grok (Collapsed):** Models exposed to severe collapse fail to develop this sharp attention concentration. Entropy remains elevated or collapses onto irrelevant artifacts, indicating failure to learn the underlying causal circuit.

*Generated Artifacts:*
- `analysis/attention/entropy_evolution.png`
- `analysis/attention/concentration_evolution.png`
- `analysis/attention/*_heatmap_step_*.png`

## Conclusion

Model collapse, introduced through synthetic recursive training, fundamentally disrupts the representational geometry required for grokking. The inability to form high-rank, robust causal circuits (as evidenced by attention entropy and weight norms) confirms that collapse is not merely sample scarcity, but an active corruption of the generalization manifold.
