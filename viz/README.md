# Visualization Suite

This directory contains scripts to generate publication-quality figures for the paper, analyzing the effects of model collapse on grokking in Transformers.

## Scripts Overview

### `attention_patterns.py`
Analyzes the inner workings of the Transformer heads.
- Extracts and visualizes attention weights across the sequence.
- Generates heatmaps comparing pure models versus collapsed models.
- Generates animated GIFs showing the evolution of attention patterns over training time.
- **Usage:** `python viz/attention_patterns.py --results-dir results --output-dir viz_output`

### `weight_norms.py`
Analyzes parameter magnitudes and trajectories.
- Extracts the $L_2$ norm of weights layer-by-layer (embeddings, projections, MLP, output head).
- Plots faceted line charts showing weight norm progression across varying collapse levels.
- Generates 3D surface plots showing norm evolution across (training_step $\times$ collapse_level).
- **Usage:** `python viz/weight_norms.py --results-dir results --output-dir viz_output`

### `loss_landscape.py`
Visualizes the geometry of the optimization landscape.
- Picks two random, orthogonal, layer-wise normalized directions in parameter space.
- Evaluates the test loss on a 2D grid around the current model checkpoint.
- Produces contour plots comparing the landscape flatness/sharpness of pure vs collapsed models across training steps.
- **Usage:** `python viz/loss_landscape.py --results-dir results --output-dir viz_output`

### `grokking_cliff.py`
Produces comprehensive phase diagrams based on grid search experiments.
- Parses output from `results/grid/` (multiple seeds across collapse fraction and severity).
- Generates heatmaps/phase diagrams of final test accuracy and grokking step.
- Generates line plots with bootstrapped confidence intervals.
- Exports to PNG and vector PDF.
- **Usage:** `python viz/grokking_cliff.py --grid-dir results/grid --output-dir viz_output`

### `dashboard.py`
Synthesizes findings into a single, comprehensive multi-panel figure for the paper.
- Combines trajectory plots, phase diagrams, norm surfaces, and landscape cross-sections.
- Automatically handles layout and formatting to match publication standards.
- **Usage:** `python viz/dashboard.py --results-dir results --viz-dir viz_output --output viz_output/dashboard_main_figure.png`
