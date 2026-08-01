# Mechanistic Analysis of Grokking and Collapse

This document explains the suite of tools built to track weight geometry, information flow, gradient dynamics, and phase transitions when a model undergoes training on synthetic data (model collapse) versus pure data (grokking).

## 1. Weight Dynamics (`src/analysis/weights.py`)

Tools to analyze how the network's weights evolve:
*   `get_layer_norms`: Tracks L1, L2, Frobenius, and spectral norms. Rapid reduction in weight norms (30-42%) is a strong signature of collapse.
*   `get_weight_distributions`: Tracks kurtosis and skewness, identifying if the network is concentrating its mass on specific features.
*   `get_effective_ranks`: Computes the Shannon entropy of normalized singular values. When a model collapses, its embedding and projection matrices often become low-rank, indicating a loss of structural complexity.

## 2. Information Flow (`src/analysis/information.py`)

*   `compute_mutual_information`: Evaluates mutual information between arrays to see how well input signals are preserved or compressed across layers.
*   `compute_cka`: Computes Centered Kernel Alignment to measure representational similarity between layers or between checkpoints over time. This helps visualize the difference in learning trajectories between grokking and collapse.

## 3. Phase Transition Detection (`src/analysis/phase_transition.py`)

*   `detect_grokking_transition`: Automatically finds the discrete jump in accuracy, which usually signals the transition from memorization to generalization.
*   `detect_collapse_onset`: Computes the second derivative of the weight norm trajectory to pinpoint the step where collapse rapidly accelerates.

*Note: Generating a complete Phase Diagram mapping collapse severity to accuracy over time is done via `src/experiments/phase_diagram.py`.*

## 4. Gradient Analysis (`src/analysis/dynamics.py`)

*   `track_gradient_norms`: Logs gradient magnitude per layer.
*   `compute_gradient_noise_scale`: Measures batch-to-batch gradient variance.
*   `estimate_hessian_eigenvalues`: Uses power iteration to find the maximum eigenvalue of the Hessian matrix. This determines the sharpness of the local minima (i.e., curvature of the loss landscape).

## 5. Experiment Reproducibility

To run the full suite of collapse severities (pure, low, medium, severe, high) across multiple seeds:
```bash
uv run python src/experiments/run_scaling.py --seeds "42,43,44" --max-steps 50000
```
