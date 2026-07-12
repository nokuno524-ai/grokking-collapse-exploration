# Model Collapse and Grokking: Comprehensive Findings

This document summarizes the findings from running experimental grids that examine the interplay between model collapse (degenerative synthetic data learning) and delayed generalization (grokking).

## 1. Overview
The experiments evaluated how replacing clean modular arithmetic data with collapsed, generated data affects the model's ability to grok on small-scale arithmetic tasks. We categorized training conditions into `pure` (clean data) and varying degrees of collapse severity (`low_collapse`, `medium_collapse`, `high_collapse`, `severe_collapse`).

## 2. Key Results

- **Grokking Cliff Identified:**
  The models exhibit a sharp grokking cliff as data collapse increases. Under pure conditions, the model reaches 100% test accuracy (grokking around step 1400). As collapse severity increases, grokking is delayed (e.g., `low_collapse` groks around step 3100) or fails entirely (`severe_collapse` stalls at near 0% test accuracy).

- **Weight Norm and Collapse Correlation:**
  A high correlation exists between collapse severity and the trajectory of the model's weight norm. Models that fail to grok exhibit weight norms that continue to grow monotonically (overfitting to the noise/collapsed labels), whereas models that grok show a sharp regularization phase where weight norm falls off.

- **Attention Mechanism Degeneration:**
  By evaluating Q/K matrix products over training, it's evident that severe collapse inhibits the formation of coherent algorithmic structures. The mutual information between Query and Key weights is significantly lower (e.g., ~0.85 nats in severe collapse vs ~2.16 nats in pure), meaning attention heads fail to diversify functionally.

- **Loss Trajectory Power-Laws:**
  The decay of the test loss prior to the grokking phase exhibits power-law behavior. As collapse increases, the exponent shifts from steep decay (~ -0.59 for pure) toward near-flat lines (~ 0.02 for severe), statistically confirming the failure to establish learning momentum toward a generalizable solution.

## 3. Conclusions
This analysis supports the hypothesis that model collapse acts similarly to targeted uniform label noise, directly preventing the phase transitions necessary for grokking. These metrics provide a quantifiable lens (via bootstrap CIs and weight trajectories) to track whether an LM exposed to synthetic data is progressing toward delayed generalization or irrecoverable overfitting.