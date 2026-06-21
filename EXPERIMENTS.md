# Grokking and Model Collapse Experiments

This document catalogs the experiments investigating the interplay between LLM model collapse and grokking on synthetic modular arithmetic tasks.

## 1. Pure Baseline Experiment
**Condition Name:** `pure`
* **Hypothesis:** A model trained on a clean synthetic dataset with sufficient weight decay will exhibit delayed generalization (grokking), experiencing a sudden jump in test accuracy long after training loss plateaus.
* **Configuration:** `collapse_level = 0.0`, `noise_fraction = 0.0`.
* **Expected Result:** Grokking occurs around step 1400. The Fourier concentration metric will spike alongside test accuracy.

## 2. Low Collapse Experiment
**Condition Name:** `low_collapse`
* **Hypothesis:** Introducing a small amount of collapsed synthetic data will delay grokking and reduce the emergence of structured representations.
* **Configuration:** `collapse_level = 0.05`, `collapse_severity = 0.3`.
* **Expected Result:** Grokking happens significantly later than the pure baseline, with diminished Fourier concentration peaks.

## 3. Medium Collapse Experiment
**Condition Name:** `medium_collapse`
* **Hypothesis:** Moderate contamination degrades the loss landscape enough that the model struggles to escape the generalization threshold cleanly.
* **Configuration:** `collapse_level = 0.15`, `collapse_severity = 0.5`.
* **Expected Result:** Grokking either fails to occur or occurs very late, demonstrating that collapse prevents delayed generalization.

## 4. Severe Collapse Experiment
**Condition Name:** `severe_collapse`
* **Hypothesis:** Heavy contamination with a severely narrowed output distribution completely destroys the underlying mathematical structure required for grokking.
* **Configuration:** `collapse_level = 0.50`, `collapse_severity = 0.9`.
* **Expected Result:** The model overfits to the noisy data and never generalizes (no grokking).

## 5. Scarcity Dissociation Control
**Condition Name:** `scarcity_baseline`
* **Hypothesis:** Model collapse is distinct from merely reducing the effective sample size.
* **Configuration:** 50% training data without corruption.
* **Expected Result:** The model still groks successfully (sometimes with even higher Fourier concentration), whereas 15% corrupted data completely prevents grokking.

## 6. Uniform Label Noise Control
**Condition Name:** `noise_baseline`
* **Hypothesis:** Random uniform label noise produces a statistically similar breakdown of grokking compared to temperature-warped collapse contamination.
* **Configuration:** `noise_fraction = 0.15` (matched to medium collapse).
* **Expected Result:** The degradation is statistically indistinguishable from the `medium_collapse` condition, demonstrating that collapse contamination acts similarly to label noise.
