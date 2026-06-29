# Experiments Guide

This document describes the phase transition experiments, mechanistic dashboard, and statistical tests added to explore the interplay between model collapse and grokking.

## 1. Phase Transition Experiments

`experiments/phase_transitions.py` runs systematic experiments varying four main axes:
- `collapse_level` (0.0 to 0.5): Fraction of training data replaced by synthetic outputs.
- `collapse_severity` (0.0 to 1.0): Temperature control over the synthetic generator's output distribution.
- `label_noise` (0.0 to 0.3): Fraction of training data uniformly corrupted.
- `weight_decay` (0.0 to 3.0): Regularization strength.

### Running

```bash
uv venv .venv && source .venv/bin/activate
export PYTHONPATH=$(pwd)
python experiments/phase_transitions.py --collapse-levels 0.0,0.15,0.30 --collapse-severities 0.0,0.5,1.0 --label-noises 0.0,0.15,0.30 --weight-decays 0.0,1.0,3.0
```

Results are saved to `results/phase_transitions/`.

## 2. Interactive Visualization Suite

`visualizations/interactive_dashboard.py` is a Streamlit application allowing researchers to explore the multi-dimensional grid of results.

It features:
- **Phase diagram explorer:** Sliders for Weight Decay and Severity, outputting a heatmap of grokking rate vs Label Noise and Collapse Level.
- **Training curves:** Select a specific seed to view train/test loss and accuracy.
- **Mechanistic timeline:** Align the grokking step with Weight Norm and Fourier Concentration.

### Running

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)
streamlit run visualizations/interactive_dashboard.py
```

## 3. Statistical Analysis

`analysis/statistical_significance.py` computes non-parametric tests (Mann-Whitney U) and effect sizes (Cohen's d) to compare the "Pure Label Noise" vs "Pure Model Collapse" settings, specifically aiming to rigorously test whether the two phenomena differ. It outputs a LaTeX table.

### Running

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)
python analysis/statistical_significance.py
```

## 4. Publication Figures

`visualizations/publication_figures.py` generates high-quality `matplotlib` and `seaborn` figures representing the core findings of the project, specifically a grid of grokking rates and test accuracies by weight decay, and a mechanistic comparison plot.

### Running

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)
python visualizations/publication_figures.py
```
