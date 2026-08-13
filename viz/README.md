# Visualization Scripts

This directory contains scripts used to analyze and visualize the behavior of the modular arithmetic transformer model under different collapse conditions.

## Setup

Make sure your environment is activated and dependencies are installed. You can run scripts using `uv run`.

## Scripts

### 1. `plot_training_curves.py`
Reads `results.json` files for different conditions (e.g., `pure`, `low_collapse`, etc.) from the `results/` directory and plots accuracy and loss trajectories over time. It also marks the point of "grokking" (where test accuracy first reaches 80%) with a vertical dashed line.
- **Input**: `results/<condition>/results.json`
- **Output**: `viz/output/training_curves.png`
- **Usage**: `uv run python viz/plot_training_curves.py`

### 2. `plot_weight_analysis.py`
Scans the PyTorch checkpoints in the `results/` directory, computes the total L2 weight norm at each saved step, and tracks the weight norm trajectories across training. It outputs a trajectory plot over steps and a final bar chart of norms per condition.
- **Input**: `results/<condition>/checkpoint_*.pt`
- **Output**: `viz/output/weight_analysis.png`
- **Usage**: `uv run python viz/plot_weight_analysis.py`

### 3. `plot_attention_patterns.py`
Loads a specific model checkpoint and analyzes its self-attention patterns. It computes the attention queries and keys manually from `in_proj_weight` and `in_proj_bias`, visualizing the attention heatmap for each head across a forward pass, and calculates the attention entropy per head.
- **Input**: `results/<condition>/checkpoint_*.pt`, `results/<condition>/results.json`
- **Outputs**: `viz/output/attention_patterns.png`, `viz/output/attention_entropy.png`
- **Usage**: `uv run python viz/plot_attention_patterns.py`
