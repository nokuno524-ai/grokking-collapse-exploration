# Mechanistic Analysis of Grokking vs. Collapse

This document summarizes the mechanistic interpretability findings and tools created during Phase 4 to investigate why model collapse prevents grokking in Transformers.

## 1. Circuit Analysis
- **Goal**: Understand if specific attention heads specialize to solve the task (forming "grokking circuits") and how collapse disrupts this.
- **Approach**: We extract attention weights (`analysis/circuits.py`) and compare patterns pre- and post-grokking. We identify specific heads that only form rigid, specialized patterns when the model groks. In collapsed models, these circuits either fail to form entirely or are disrupted by noise.

## 2. Weight Space Analysis
- **Goal**: Track structural changes in model parameters over time.
- **Approach**: Using `analysis/weight_space.py`, we monitor the effective rank of embeddings, total weight norm, and Hessian eigenvalues. The effective rank provides a measure of representation dimensionality; grokking is characterized by a sharp drop in effective rank (compression). Collapsed models tend to maintain higher rank, suggesting a failure to compress representations.

## 3. Gradient Flow Analysis
- **Goal**: Determine if learning stalls due to gradient issues.
- **Approach**: True gradients aren't stored in checkpoints, so we approximate them via consecutive weight updates (`W_t - W_{t-1}`). Using `analysis/gradient_flow.py`, we identify "gradient starvation" where specific parameter updates shrink dramatically under collapse compared to pure data training. We also measure gradient noise scale to study learning dynamics.

## 4. Interventions
- **Goal**: Move from observational analysis to causal understanding.
- **Approach**: The `experiments/interventions.py` module introduces tools to ablate specific attention heads (zeroing their output projections) and to freeze specific layers during training. These tools can prove that the specific "grokking circuits" found observationally are indeed responsible for the task performance.
