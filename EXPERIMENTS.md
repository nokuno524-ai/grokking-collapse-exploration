# Extended Experiments: Grokking vs Model Collapse

This document describes the new experiments that extend the initial research into how model collapse impacts the grokking phenomenon in transformer models.

## 1. Model Scaling Sweep
- **Script:** `experiments/scaling.py`
- **Config:** `configs/scaling.yaml`
- **Description:** Runs a comprehensive sweep over model depths (1-6 layers), widths (64-512 dims), and collapse levels (0.0 - 0.7). It saves the training results (including grokking steps and final test accuracies) into a structured JSON summary format. This allows us to observe whether increasing model capacity can mitigate the anti-grokking effect of model collapse, and to fit Chinchilla-style scaling laws.

## 2. Curriculum Learning
- **Script:** `experiments/curriculum.py`
- **Config:** `configs/curriculum.yaml`
- **Description:** Evaluates if the anti-grokking effect of contaminated data can be avoided by strategically altering the dataset composition during training.
- **Schedules tested:**
  - `linear`: Gradually increase contamination fraction from `start` to `end`.
  - `step`: Pure data for the first N% of steps, then fully contaminated.
  - `reverse`: Start heavily contaminated, gradually purify.
  - `random`: Random contamination fraction at each step.
  - `constant`: Control condition.

## 3. Data Quality Threshold
- **Script:** `experiments/threshold.py`
- **Config:** `configs/threshold.yaml`
- **Description:** Performs a binary search over `collapse_level` (from 0.0 to 0.5) to identify the exact phase transition threshold at which the model completely stops grokking. Runs across multiple seeds to establish robust statistical confidence intervals (via bootstrap).

## 4. Extended Analysis
- **Script:** `analysis/extended.py`
- **Description:** Centralized analysis script that:
  - Fits scaling law coefficients (width, depth, collapse level effects) to the output of the scaling sweep.
  - Evaluates the success rates and grokking delays of different curriculum schedules.
  - Computes the mean critical threshold and 95% bootstrap confidence intervals for the model collapse phase transition.
