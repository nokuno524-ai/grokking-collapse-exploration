# Extended Experiment Catalog

This document tracks the new experiment configurations introduced in our deeper analysis phase.

## 1. Partial Collapse (Data Mixing)
**Goal:** Determine if there is a tipping point where synthetic data prevents grokking, even when pure data is present.
- **Config:** `DatasetConfig(collapse_level=0.0)` mixed with `collapse_level=1.0`.
- **Ratios Tested:** 10%, 25%, 50%, 75% synthetic data.
- **Driver Logic:** `setup_partial_collapse_experiment()` in `experiments/extensions.py`.

## 2. Recovery (Curriculum)
**Goal:** Determine if a collapsed model can recover and grokk when presented with clean data, or if the "curse" is permanent in the weights.
- **Setup:** Load `results/high_collapse/checkpoint_50000.pt`.
- **Intervention:** Replace the dataloader with a purely clean `collapse_level=0.0` distribution.
- **Driver Logic:** `setup_recovery_experiment()` in `experiments/extensions.py`.

## 3. Transfer Learning
**Goal:** Test if grokked representations (e.g., modular arithmetic circuits) transfer to new, but related tasks.
- **Setup:** Train on mod 59 (`prime=59`) until grokking.
- **Intervention:** Swap the embedding and output head to size 61 (`prime=61`), freeze the transformer core, and retrain on mod 61.
- **Driver Logic:** `setup_transfer_experiment()` in `experiments/extensions.py`.

## 4. Scaling Laws
**Goal:** Assess how the collapse-grokking threshold changes as a function of model capacity.
- **Config Sweeps:**
  - `d_model`: [64, 128, 256, 512]
  - `n_layers`: [1, 2, 4]
  - `n_heads`: [2, 4, 8]
- **Hypothesis:** Wider models may delay the point of total collapse, but the phase transition remains sharp.
- **Driver Logic:** `setup_scale_experiment()` in `experiments/extensions.py`.
