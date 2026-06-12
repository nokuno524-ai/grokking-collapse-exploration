# Mechanistic Analysis Pipeline

This directory contains tools for mechanistically analyzing models trained in the context of the Grokking vs Model Collapse studies.

## Components

### 1. Weight Analysis (`weight_analysis.py`)
Computes various norms and the effective rank for the model weights over the course of training.

**Mathematical Formulations:**
- **L1 Norm**: $\|W\|_1 = \sum_{i,j} |W_{i,j}|$
- **Frobenius Norm**: $\|W\|_F = \sqrt{\sum_{i,j} W_{i,j}^2}$
- **Spectral Norm (L2)**: $\|W\|_2 = \sigma_{max}(W)$
- **Effective Rank**: Evaluates the Shannon entropy of normalized singular values: $H = -\sum \tilde{\sigma}_i \ln(\tilde{\sigma}_i)$, Rank = $\exp(H)$, where $\tilde{\sigma}_i = \frac{\sigma_i}{\sum \sigma_j}$.

**Usage Example:**
```python
from analysis.weight_analysis import analyze_checkpoints, plot_weight_evolution

checkpoints = ["ckpt_100.pt", "ckpt_200.pt"]
results = analyze_checkpoints(checkpoints)
plot_weight_evolution(results, metric="frobenius", save_path="weight_evolution.png")
```

### 2. Circuit Discovery (`circuit_discovery.py`)
Provides hooks for resample ablation to analyze the importance of individual attention heads.

**Technique:**
Activation patching (resample ablation) swaps the activation of a specific attention head in a clean model pass with the activation from a corrupted/baseline pass.
The importance score is then evaluated based on the logit difference.

**Usage Example:**
```python
from analysis.circuit_discovery import generate_importance_heatmap
import numpy as np

# Suppose we obtained scores from compute_logit_diff over all heads and layers
scores = np.random.rand(1, 4)  # 1 layer, 4 heads
generate_importance_heatmap(scores, save_path="importance.png")
```

### 3. Representation Analysis (`representation.py`)
Measures similarity between model representations over time using Centered Kernel Alignment (CKA) and calculates the effective rank of representation spaces.

**Mathematical Formulations:**
- **Linear CKA**: $CKA(X, Y) = \frac{\|X_c^T Y_c\|_F^2}{\|X_c^T X_c\|_F \|Y_c^T Y_c\|_F}$ where $X_c$ and $Y_c$ are centered representations.

**Usage Example:**
```python
from analysis.representation import compute_all_pairs_cka, plot_cka_matrix
import numpy as np

reps = [np.random.randn(100, 128) for _ in range(5)]
cka_matrix = compute_all_pairs_cka(reps)
plot_cka_matrix(cka_matrix, step_labels=["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"], save_path="cka_matrix.png")
```
