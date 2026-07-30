# Deep Analysis Tooling Guide

This guide details the extended analysis pipeline and tooling established for analyzing the intersection of model collapse and grokking within the training dynamics of transformer architectures.

## Overview

The analysis tools live under the `src/analysis/` module. They cover aggregating multi-condition results, conducting mechanistic circuit explorations (causal attribution), and computing rigorous statistical measures.

### 1. Results Aggregation (`src/analysis/parse_results.py`)

Provides tools to recursively scrape all `results.json` files generated from independent experiment loops.

- **Functionality**: Extracts step trajectories, grokking milestones, architecture parameters, collapse levels, and training conditions.
- **Key Method**: `aggregate_results(results_dir: str) -> pd.DataFrame` produces a centralized, Pandas-friendly dataset, merging across experimental setups (e.g., pure, low collapse, medium, high). It ignores intermediate directories like grid sweeps or seeds for cleaner high-level data aggregation.

### 2. Mechanistic Interventions (`src/analysis/circuits.py`)

A set of causal discovery and interpretability methods for probing internal transformer representations.

- **Logit Attribution (`get_logit_attribution`)**:
  Deconstructs output logits into discrete additive influences from different components (embeddings, attention heads, and MLP outputs) along the residual stream. Use this to determine exactly *when* the attention blocks specialize into solving the modular arithmetic task.

- **Activation Patching (`activation_patching`)**:
  Causally isolates sub-networks. This method swaps (patches) intermediate activations—like embedding vectors, attention layer outputs, or MLP transformations—from a "grokked" model into an equivalent "collapsed" (corrupted) model to isolate the recovery point where functional representation emerges.

- **Integrated Gradients for Attention (`integrated_gradients_attention`)**:
  Computes precise importance scoring for multi-head attention components using gradient-based attribution integrated over interpolations from a zero baseline to the actual network activation state.

### 3. Statistical Analysis (`src/analysis/statistics.py`)

Provides rigorous validation for grokking delay points and feature correlations.

- **Bootstrap Confidence Intervals (`bootstrap_confidence_interval`)**:
  Quantifies uncertainty around critical transition points (like mean grokking steps) using Monte-Carlo resampling of experimental repeats to derive 95% CIs.

- **Multiple Regression & Correlations (`analyze_grokking_factors`)**:
  Uncovers statistical dependencies linking feature shifts (such as total weight norm decreases, embedding dimension effective ranks, and label collapse severities) with grokking success criteria using OLS multiple regression. Missing transition values (for non-grokking models) are automatically handled via substitution mapping them to `max_steps`.

## Typical Workflow

1. Train initial models using standard run scripts (`src/train.py`, etc.).
2. Accumulate outputs under `results/`.
3. Read the global landscape using `df = aggregate_results()`.
4. Validate model performance bounds and correlation impacts with `analyze_grokking_factors(df)`.
5. Drill into failure modes of non-grokked models via `activation_patching` matched against "pure" checkpoints.

## Testing Setup
To run tests over the analytical components, ensure you have correctly installed the required environments listed in the setup blocks and run:
```bash
uv run pytest tests/test_analysis.py
```
This guarantees core mechanisms remain stable as future causal interpretations evolve.
