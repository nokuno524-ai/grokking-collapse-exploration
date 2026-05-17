# Phase Analysis Suite

This document outlines the analysis modules added to monitor and identify key phases during model training, specifically tailored to understanding grokking and model collapse.

## Weight Analysis (`src/analysis/weight_analysis.py`)

The weight analysis module provides tools to track the evolution of network parameters:

- **Weight Norms:** Computes L2 norms (`get_weight_norms`) for individual parameters and the entire network to track parameter growth, a known correlate of grokking.
- **Effective Rank:** Calculates the effective rank (`get_effective_rank`, `get_layer_effective_ranks`) of weight matrices using the Shannon entropy of their singular value distribution. This indicates whether the network is collapsing into a low-dimensional subspace.
- **Singular Value Distribution:** Extracts the singular values of layers (`get_singular_value_distribution`).
- **Weight Velocity:** Measures the L2 distance between models at different timesteps (`calculate_weight_velocity`), representing how fast the parameters are moving through the loss landscape.

## Gradient Dynamics (`src/analysis/gradient_dynamics.py`)

The gradient dynamics module provides insights into the optimization process:

- **Gradient Norms:** Computes L2 norms (`get_gradient_norms`) for individual parameter gradients and the total gradient to monitor optimization stability.
- **Gradient Noise Scale:** Estimates the variance of gradients across mini-batches (`estimate_gradient_noise_scale`), useful for identifying periods where the signal-to-noise ratio drops.
- **Gradient Coherence:** Calculates cosine similarity between consecutive gradient steps (`calculate_gradient_coherence`), indicating whether the optimizer is moving consistently or oscillating.
- **Vanishing/Exploding Detection:** Automatically flags parameters whose gradients cross user-defined thresholds (`detect_gradient_vanishing_explosion`).

## Phase Transition Detection (`src/analysis/phase_detector.py`)

The phase transition module operates on the metrics extracted by the other suites to automate the detection of important training events:

- **Grokking Transition:** Identifies the critical step where test accuracy spikes (`detect_grokking_transition`), indicating the network has generalized.
- **Collapse Onset:** Detects sudden explosions in weight norms (`detect_collapse_onset`) relative to historical rolling averages, indicating the onset of optimization failure or model collapse.
- **Intervention Periods:** Finds contiguous training steps (`identify_intervention_periods`) where a specific metric (e.g., gradient noise) is persistently above or below a threshold. This can be used to trigger dynamic interventions like increasing weight decay or altering the learning rate.

## Integration

These modules are designed to be integrated into the training loop (`src/train.py`) or called post-hoc during artifact evaluation. They rely on standard PyTorch hooks or simple state evaluation over the model.
