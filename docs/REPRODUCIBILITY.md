# Reproducibility Checklist

This document provides a comprehensive guide to reproducing the results in the **Grokking Cliffs** repository.

## Environment Setup

The repository uses `uv` for fast dependency management. We recommend creating a virtual environment.

```bash
# Initialize and activate the virtual environment
uv venv .venv
source .venv/bin/activate

# Install the required Python packages
uv pip install -r requirements.txt
# OR install directly
uv pip install torch numpy matplotlib scipy pandas jinja2
```

> Note: If you encounter issues compiling loss landscape videos, ensure `ffmpeg` is installed on your system (e.g., `sudo apt-get install ffmpeg` on Ubuntu).

## Random Seed Handling

To ensure reproducible runs, we fix random seeds across PyTorch and NumPy at the start of training and dataset generation:

- The default seed is `42`.
- In grid search and repeat experiments (e.g., `run_multi_seed.py`), we iterate through seeds: `42, 43, 44, 45, 46`.
- We use the `torch.manual_seed(seed)` and `np.random.RandomState(seed)` pattern to ensure weight initialization, dataset sampling, and data loader shuffling are deterministic.

## Experiment Hyperparameters

The default hyperparameters for the `pure` condition (which groks reliably) are:

- **Architecture:** 1-layer Transformer encoder (`ModularArithmeticTransformer`)
- **Task:** `(a + b) mod p` where `p = 59`
- **Parameters:** ~214K (`d_model=128`, `n_heads=4`, `d_ff=512`)
- **Optimizer:** AdamW
- **Learning Rate:** `1e-3`
- **Weight Decay:** `1.0`
- **Batch Size:** `512`
- **Max Steps:** `50,000`
- **Train Fraction:** `0.3` (30% of the possible pairs)
- **Collapse Level:** `0.0`
- **Collapse Severity:** `0.5`

For collapse conditions, the `collapse_level` changes (e.g., `0.05` for low collapse, `0.3` for high collapse, `0.5` for severe collapse).

## Expected Results

Based on our experiment outputs in `results/`, you should expect the following behavior:

1.  **Pure Condition (`pure`)**:
    *   **Grokking Step:** ~1,400 to 1,500.
    *   **Final Test Accuracy:** 1.0 (100%).
    *   **Behavior:** The model memorizes the training data quickly, then after a delay, test accuracy rapidly spikes to 1.0.

2.  **Low Collapse (`low_collapse`)**:
    *   **Grokking Step:** ~2,700 to 3,100 (delayed compared to pure).
    *   **Final Test Accuracy:** ~0.97.
    *   **Behavior:** Grokking still occurs but is noticeably delayed and slightly degraded.

3.  **Medium Collapse (`medium_collapse`)**:
    *   **Grokking Step:** N/A (does not reach 95% test accuracy within 50k steps).
    *   **Final Test Accuracy:** ~0.85.
    *   **Behavior:** The model overfits but fails to fully generalize, entering a degraded generalization regime.

4.  **High/Severe Collapse (`high_collapse`, `severe_collapse`)**:
    *   **Grokking Step:** N/A.
    *   **Final Test Accuracy:** < 0.35 for high, < 0.05 for severe.
    *   **Behavior:** Complete failure to generalize. The model memorizes the corrupted training data and grokking is entirely suppressed.

## Running the Verification Pipeline

To parse results and generate statistical summaries:

```bash
# 1. Parse JSON logs to CSV and JSON
python src/parse_results.py results/

# 2. Run statistical tests (Confidence intervals, Mann-Whitney U)
python src/statistical_analysis.py

# 3. View the generated tables
cat analysis/statistical_summary.tex
```
