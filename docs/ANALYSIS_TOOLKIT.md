# Mechanistic Analysis Toolkit

This document describes the mechanistic analysis tools available in this repository to study grokking and model collapse in modular arithmetic transformers.

## 1. Fourier Circuit Analysis (`analysis/fourier_circuits.py`)
**Theoretical Basis**:
Following Chen et al.'s findings on stratified Fourier mechanisms, modular addition/multiplication circuits often learn representations in the Fourier domain. The 2D Fourier transform over the input grid $(a, b)$ reveals the active frequencies the model relies on.
**Usage**:
- `get_2d_fourier_transform(grid)`: Computes the 2D FFT of the $p \times p$ attention weights to identify dominant frequencies.
- `analyze_fourier_circuits(...)`: Extracts these components for a specific checkpoint.
- `plot_fourier_heatmaps(...)`: Generates heatmaps tracking the magnitude of dominant Fourier components over training.
- `compare_runs(...)`: Contrasts the Fourier spectra of grokked versus collapsed models.

## 2. Attention Pattern Evolution (`analysis/attention_evolution.py`)
**Theoretical Basis**:
D'Angelo et al. identified n-gram interpolation mechanisms in induction heads. Here, we track attention entropy and context-matching to determine when attention heads specialize. Low entropy indicates a specialized head focusing on specific inputs or positional relationships.
**Usage**:
- `get_attention_entropy(...)`: Measures how sharp the attention distributions are over inputs.
- `classify_head_context_matching(...)`: Determines if heads attend to specific relative positions (e.g., attending to position 0 from position 1).
- `analyze_attention_evolution(...)`: Plots specialization timelines, helping pinpoint critical transition points where the model reorganizes its circuits.

## 3. J-Space Probe (`analysis/j_space_probe.py`)
**Theoretical Basis**:
Inspired by Anthropic's Jacobian-lens (J-lens) analysis, we can identify global workspace structure by computing the Jacobian of the logits with respect to residual stream activations. Performing SVD on this Jacobian matrix identifies the highest-variance J-space directions.
**Usage**:
- `get_j_space_svd(...)`: Identifies high-variance J-space directions.
- `causal_intervention_j_space(...)`: Tests the causal importance of J-space by ablating the top-$k$ directions and measuring accuracy impact.
- `compare_j_space(...)`: Analyzes if the J-space dimensionality and structure differ between grokked and collapsed checkpoints.
