# Detectability of Synthetic Data under Model Collapse

This document summarizes the findings from our data-detection experiments (`scripts/detect_synthetic.py`), investigating whether synthetic/corrupted data that leads to model collapse can be identified *prior* to training failure (e.g. before the grokking cliff).

## Experimental Setup

We evaluated three methods for classifying a training example as real or synthetic:
1. **Learned Probe (Logistic Regression)**: A logistic regression model trained on frozen features `[h, target_embed]` from the transformer at various training checkpoints.
2. **Baseline A (Loss per example)**: Using the cross-entropy loss from the model.
3. **Baseline B (Target frequency)**: Since the synthetic distribution narrows to favor common targets, this baseline uses the frequency of the target in the training set as the only feature.

We tracked AUROC (Area Under the Receiver Operating Characteristic curve) for these detectors across training steps for `low_collapse` (5% collapse), `medium_collapse` (15% collapse), and `severe_collapse` (50% collapse).

## Findings

1. **Synthetic data is trivially detectable purely from the target distribution**
   - **Baseline B (Target frequency)** achieves an extremely high AUROC (e.g., ~0.85+ for severe collapse) completely independent of the model state or training step. Because the collapse mechanism inherently skews the label distribution towards common targets, any simple frequency-based filter can identify synthetic examples with high confidence.

2. **Model Loss quickly aligns with the collapsed distribution**
   - **Baseline A (Loss)**: Initially, the model has no knowledge of the distribution, so AUROC is ~0.5. As training progresses, the model fits the dataset, and its loss structure shifts. Collapsed examples, being highly frequent targets, tend to have *lower* loss, making loss a moderate inverse predictor of synthetic status.

3. **The Learned Probe performs marginally better than pure loss but worse than simple frequency**
   - Using the model's internal representations `h` alongside the target embedding allows the probe to partially learn the corrupted mapping. However, because the synthetic contamination purely affects the *label distribution* rather than the input-label relationship (inputs are uniformly sampled, but labels are replaced with high-frequency outputs), the model representations don't offer much more insight than simple statistics.

## Intervention Proposal (Future Work)

Because simple target frequency statistics are such a strong predictor of synthetic corruption in this setting, early data-filtering could completely prevent model collapse.

A sketched intervention:
1. **Compute Target Frequencies**: Before training, compute the histogram of labels in the training set.
2. **Filter Outliers**: Drop examples whose label frequencies exceed a threshold (e.g., >2 standard deviations above the mean for uniform tasks like modular arithmetic).
3. **Train on Filtered Data**: By truncating the head of the label distribution, we remove the synthetic contamination. As our prior scarcity baselines show, the model can still grok effectively with less data, as long as the remaining data is clean.

This suggests that for synthetic data exhibiting basic collapse signatures (distribution narrowing), cheap, purely statistical filters are highly effective early-warning mechanisms, precluding the need for complex activation-based probes.
