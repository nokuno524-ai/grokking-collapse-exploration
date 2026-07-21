# Phase Transitions and Grokking Dynamics

This document details the analytical frameworks added to characterize grokking as a phase transition and understand its training dynamics, predictability, and scaling laws.

## Information-Theoretic Analysis

The transition from memorization to generalization in grokking can be viewed through an information-theoretic lens.
* **Mutual Information (MI)**: We track the mutual information between internal model representations and the task labels.
* **Information Bottleneck**: The delayed generalization phase (grokking) corresponds to a compression phase where the model discards task-irrelevant information while maintaining or improving task-relevant information.

## Phase Transition Perspective

Grokking is characterized by a sharp phase transition in test accuracy.
* **Order Parameters**: We define the generalization gap, gradient norm proxy, and loss curvature as order parameters that signify the phase shift.
* **Sigmoid Fitting**: The test accuracy trajectory is fit to a sigmoid curve `L / (1 + exp(-k*(x-x0)))`. We extract:
  * `k`: The sharpness of the transition.
  * `x0`: The midpoint of the transition (grokking step).
* By modeling the test accuracy with these equations, we can quantitatively compare the effect of different collapse and noise conditions on the transition dynamics.

## Scaling Laws

We establish empirical scaling relationships for the grokking step (`S_grok`).
* **Power Laws**: We fit `S_grok ~ a * X^b` where `X` is the model size or data size.
* **Predictive Extrapolation**: These derived scaling laws allow predicting when (or if) larger models will grok under specific conditions.

## Early Training Dynamics & Predictability

Can we predict grokking from early training signals?
* **Landscape Sharpness**: We monitor the trace of the Hessian (approximated via gradient variances) and weight movement velocities.
* **Predictive Modeling**: By feeding early trajectory features (loss slope, gradient noise variance, train-test gaps) into a Logistic Regression classifier, we can predict the occurrence of grokking well before the phase transition happens.
