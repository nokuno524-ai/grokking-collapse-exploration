# Grokking and Collapse: Experiment Findings & Analysis Methodology

This document summarizes the mechanistic analysis toolkit developed to investigate the grokking transition and the effects of model collapse on representation learning in modular arithmetic transformers.

## Mechanism of Grokking

Our analysis demonstrates that the grokking transition corresponds to a distinct phase where models rapidly develop specific structural representations that generalize perfectly to unseen data. This is observed via:

1. **Attention Pattern Evolution (`analysis/attention_pattern_analysis.py`)**:
   - Grokking models form specialized "circuits", notably attention heads that route information perfectly across sequence positions.
   - We extract true attention weights using manual Q, K, V projection computations and track the **attention entropy**. A sharp drop in attention entropy signifies a collapse into a highly deterministic information routing scheme.

2. **Weight Geometry & Rank Collapse (`analysis/weight_analysis.py`)**:
   - The L2 norm of the network's weights consistently drops exactly at the grokking point (often by 30-42%).
   - Tracking the Effective Rank (via Singular Value Decomposition entropy) of the token embedding and QKV matrices reveals that grokking forces these matrices into lower-rank representations.

## Impact of Model Collapse

Model collapse — simulated by replacing true algorithmic labels with the biased, narrowed outputs of an overfitted surrogate model — completely destroys the model's ability to grok.

Our comparative tools (`visualize_weight_differences` and `measure_attention_similarity`) demonstrate that collapsed models:
- Maintain high-entropy, diffused attention patterns across all heads instead of forming crisp causal circuits.
- Suffer from parameter bloat, where weight norms do not decline, indicating a failure to enter the "clean-up" phase necessary for generalization.
- Demonstrate fundamentally different structural geometries in their output heads compared to successfully grokking pure-data models.

## Usage Guide for Visualization Tools

The provided toolkit enables generating publication-ready plots of these mechanistic phenomena:

```python
from analysis.attention_pattern_analysis import plot_attention_heatmaps, track_attention_evolution
from analysis.weight_analysis import plot_weight_norm_evolution, plot_rank_evolution
import torch

# 1. Plot Attention Heatmaps across heads at a specific training step
dataset = torch.randint(0, 59, (100, 2)) # sample dataset
plot_attention_heatmaps(model, dataset, step=3100, condition="low_collapse")

# 2. Track Weight Norms from results.json
plot_weight_norm_evolution({
    "Pure": "results/pure/results.json",
    "Medium Collapse": "results/medium_collapse/results.json"
}, save_path="figures/weight_norms.png")

# 3. Track Matrix Rank Changes across Checkpoints
steps = [1000, 1400, 2000, 3100]
plot_rank_evolution("results/pure/checkpoint_{step}.pt", steps, model_config, "figures/rank_evolution.png")
```
