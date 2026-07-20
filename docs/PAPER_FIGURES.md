# Publication Figures

This directory contains scripts used to generate the final publication-ready figures for the paper "Contamination Cliffs: Causal Mechanism and Threshold Theory for Grokking Failure under Label Noise".

## Figures

### Figure 1: Experimental Setup (`figures/fig1_setup.py`)
**Description:** Diagrams the data generation pipeline and the `ModularArithmeticTransformer` architecture. Shows how the baseline dataset is either corrupted via random label noise or temperature-warped sampling to simulate model collapse.
**Output:** `figures/fig1_setup.png` / `.pdf`
**Caption:** *Experimental setup for evaluating grokking under label noise and model collapse contamination. The 1-layer transformer is trained on modular addition, with the training set subjected to either true label noise or synthetic data contamination.*

### Figure 2: Main Results (`figures/fig2_main_results.py`)
**Description:** Plots the training and test accuracy trajectories across pure, low collapse, and severe collapse conditions. Shaded regions represent $\pm 1$ standard deviation across 5 random seeds.
**Output:** `figures/fig2_main_results.png` / `.pdf`
**Caption:** *Test and train accuracy over training steps. While the pure and low-collapse (5%) models eventually grok (achieve $>90\%$ test accuracy), severe collapse (30%) entirely prevents grokking despite full training accuracy.*

### Figure 3: Mechanistic Analysis (`figures/fig3_mechanism.py`)
**Description:** A side-by-side view of three key geometric and mechanistic metrics over training: Total weight $L_2$ norm, Token embedding effective rank, and Fourier concentration.
**Output:** `figures/fig3_mechanism.png` / `.pdf`
**Caption:** *Mechanistic signatures of grokking versus collapse. The pure model undergoes a phase transition characterized by a reduction in weight norm, a collapse in embedding rank (increasing structure), and sharp Fourier concentration. Severe contamination suppresses these cleanup phase dynamics.*

### Figure 4: Phase Diagram (`figures/fig4_phase_diagram.py`)
**Description:** Visualizes the grokking phase transition across a grid of collapse severities (temperature warping) and collapse levels (fraction of training data replaced). Shows both the probability of grokking and the mean grokking step.
**Output:** `figures/fig4_phase_diagram.png` / `.pdf`
**Caption:** *Grokking phase diagram. The collapse level (x-axis) sharply dictates whether the model groks, while the severity of the corruption (y-axis) has negligible effect once the data is sufficiently out-of-distribution.*

### Attention Evolution (Multi-panel) (`analysis/attention_evolution.py`)
**Description:** Visualizes the evolution of attention patterns (heatmaps) across early, mid, and late training stages for both pure and collapsed models.
**Output:** `results/attention_evolution.png` / `.pdf`
**Caption:** *Evolution of attention patterns over training. The pure model develops highly structured attention heads (e.g., attending to specific relative positions), while the severe-collapse model fails to form these precise circuits.*

## Generating Figures
To regenerate all figures:
```bash
python figures/fig1_setup.py
python figures/fig2_main_results.py
python figures/fig3_mechanism.py
python figures/fig4_phase_diagram.py
python analysis/attention_evolution.py
```