# Visualizations

This directory contains scripts for generating publication-quality visualizations for the experiments tracking model collapse and grokking.

## Requirements

The scripts require the virtual environment to be set up:
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt # Note: uses standard data stack (matplotlib, numpy, torch)
```

## Available Scripts

### 1. Training Curves (`plot_training_curves.py`)

Parses `results/*/results.json` files and creates subplots for test accuracy, loss, and weight norm across different collapse levels.

```bash
python visualizations/plot_training_curves.py
```
Output: `visualizations/training_curves.png`

### 2. Phase Diagram (`plot_phase_diagram.py`)

Generates a 2D plot mapping collapse severity (x-axis) to final accuracy (y-axis) showing the grokking threshold.

```bash
python visualizations/plot_phase_diagram.py
```
Output: `visualizations/phase_diagram.png`

### 3. Attention Heatmaps (`plot_attention_heatmaps.py`)

Extracts attention weights from a given model checkpoint and plots heatmaps on test sequences.

```bash
python visualizations/plot_attention_heatmaps.py --checkpoint results/pure/checkpoint_50000.pt
```
Output: `visualizations/attention_heatmaps.png`

### 4. Training Animation (`animate_training.py`)

Generates a `.gif` or `.mp4` showing the evolution of metrics over training steps.

```bash
python visualizations/animate_training.py --condition pure
```
Output: `visualizations/training_animation.gif`
