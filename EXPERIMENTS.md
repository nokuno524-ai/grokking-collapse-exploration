# Experiments Documentation

This document outlines the experiments designed to understand the scaling behavior of the interplay between model collapse and grokking, as well as the new metrics and automated detection systems introduced.

## Key Finding

**Model collapse fundamentally prevents grokking.** In pure models (0% collapse), grokking consistently occurs around step 1400, reaching 100% test accuracy. However, as the synthetic data contamination (collapse level) increases, the grokking step is delayed, and under severe collapse conditions, grokking fails entirely. The scaling experiments aim to determine if scaling model size or dataset size can mitigate this failure.

## 1. Scaling Experiments (`experiments/scaling.py`)

We systematically study the interaction between scale and collapse tolerance across three dimensions.

### Configurations

*   **Model Size (`d_model`)**: `[32, 64, 128, 256, 512]`
    *   *Hypothesis*: Larger models might resist collapse better by dedicating subsets of parameters to fit the collapsed distribution while preserving the underlying generalizing circuits.
*   **Dataset Size (`prime` p)**: `[29, 59, 97, 113, 127]`
    *   *Hypothesis*: Larger primes (more complex tasks) may be more sensitive to synthetic data contamination due to the increased difficulty of discovering the generalizing mechanism.
*   **Collapse Severity**: `[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]` (10 granular levels)
    *   *Hypothesis*: The transition from "delayed grokking" to "complete failure" is sharp (a cliff), and this transition point shifts predictably with scale.

### Execution and Expected Runtimes

*   **Total Runs**: 5 (models) × 5 (primes) × 10 (collapse levels) = 250 experimental runs.
*   **Expected Runtime**: ~2-4 hours on a modern GPU depending on early stopping behavior (some runs fail to grok and hit `max_steps`).

### Expected Results

The output consists of JSON logs mapping configurations to grokking success/steps, and scaling law plots (`scaling_law_p*.png`). We expect the plots to reveal the critical collapse threshold for each scale.

## 2. Novel Collapse Metrics (`src/collapse_metrics.py`)

To move beyond simple test accuracy, we track five continuous metrics that correlate with representation degradation:

1.  **Representation Collapse Score**: Computes the effective rank of hidden representations using Singular Value Decomposition (SVD). Lower scores mean representations are collapsing into a lower-dimensional subspace.
2.  **Gradient Collapse Score**: Tracks the cosine similarity of gradients across different training samples. High similarity indicates gradients are aligning, a sign of output collapse.
3.  **Output Diversity Index**: Measures the entropy of the output marginal distribution across different inputs. Low entropy indicates the model is predicting a narrow set of frequent classes (a hallmark of model collapse).
4.  **Weight Matrix Conditioning**: Tracks the condition number (ratio of max/min singular values) of weight matrices to detect numerical instability.
5.  **Attention Pattern Collapse**: Measures how attention patterns converge across different inputs using variance across the batch. High values indicate the attention mechanism is acting homogeneously regardless of the input tokens.

## 3. Phase Transition Detection (`src/phase_detection.py`)

We implement automated change-point detection on test accuracy curves to precisely define the phases of training under contamination.

1.  **Grokking Phase (`detect_grokking_phase`)**: Identified as the first step where test accuracy exceeds 95% and remains above that threshold for a sustained window.
2.  **Collapse Onset (`detect_collapse_onset`)**: Identified as the point where performance starts degrading significantly (e.g., >10% drop) compared to a pure baseline run.
3.  **Critical Points (`compute_critical_points`)**: Aggregates the metrics to determine the *collapse threshold* (the maximum severity level before grokking completely fails) and the *recovery potential* (how much the model can bounce back after initial collapse).
