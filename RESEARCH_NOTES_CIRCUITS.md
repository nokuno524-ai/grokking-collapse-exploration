# Circuit Analysis: Mechanistic Understanding of Collapse vs. Grokking

This document summarizes the methodology and goals behind the circuit-level analysis tooling introduced in `src/analysis/circuit_analysis.py`. These tools provide the necessary primitives to investigate *why* label-noise and weight decay cliffs occur in transformer models training on modular arithmetic.

## Goal: Mechanistic Analysis at the Circuit Level

Our previous experiments demonstrated significant cliffs when varying label noise and weight decay. However, those findings were macroscopic—looking at aggregate measures like test accuracy and Fourier concentration.

To definitively separate the effect of noise and decay, we must observe how the internal computational subgraphs (circuits) are structured. The crucial question is: **Why does a "collapsed" model (e.g., from high noise) fail to form grokking circuits, and does weight decay prevent these circuits from forming, or simply make them less robust to noise?**

## The Tooling (`src/analysis/circuit_analysis.py`)

We have introduced four key tools for mechanistic analysis:

### 1. Manual Multi-Head Attention (`manual_transformer_forward`)
PyTorch's native `nn.MultiheadAttention` is highly optimized and fused, making it difficult to perform surgical interventions (like ablating a single head). We implemented an exact, mathematically equivalent manual forward pass that can replace the native forward pass during evaluation, exposing the computation of $Q, K, V$ and context vectors.

### 2. Circuit Discovery and Ablation (`CircuitDiscoveryTool`)
With the manual attention block, we can apply specific masks to the context vector output of each head.
- **Head Ablation:** By zeroing out a head's output and measuring the performance drop, we can quantify its "importance."
- **Comparing Phases:** By running this tool on checkpoints from pre-grokking, post-grokking, and collapsed models, we can map how reliance shifts across attention heads as phase transitions occur.

### 3. SVD Component Analysis (`WeightDecomposition`)
Weight matrices (especially embedding matrices, out-projections, and FFN layers) often exhibit low-rank structure when grokking occurs (evidenced by Fourier concentration).
- We use SVD to extract the top-$k$ singular components of these matrices.
- The `compare_singular_spaces` function (based on principal angles) allows us to quantify how similar the "grokking components" (from pure models) are to the components learned by collapsed or regularized models.
- If weight decay merely limits noise tolerance, we would expect a high-WD model and a low-WD model to share a highly overlapping singular space prior to the noise cliff.

### 4. Headless Visualizations
The framework includes plotting functions built on `matplotlib` and `seaborn` (configured with the `Agg` backend) to generate:
- Attention probability heatmaps.
- Head importance matrices.
- Singular value spectrum curves.

## Findings: Applying Analysis to Pure vs Collapsed Models

Running `src/run_circuit_analysis.py` across pre-grokking (step 5000), transition (step 15000), and post-grokking (step 50000) checkpoints yields several key mechanistic insights:

### Attention Head Importance

- **Pre-Grokking (Step 5000):** Both the `pure` and `high_collapse` models exhibit distributed, uniform attention head importance. Ablating any single head leads to similar, marginal increases in loss, suggesting the network relies on memorization distributed across all heads without specialized computation.
- **Transition & Post-Grokking (`pure` at step 15000 & 50000):** As grokking occurs, the importance scores become starkly differentiated. Specific heads take on massive responsibility for the overall performance (indicated by huge loss spikes when ablated). The model forms dedicated, sparse "grokking circuits".
- **Collapsed (`high_collapse` at step 15000 & 50000):** The collapsed model never forms these specialized circuits. Head importance remains uniform and relatively low throughout training. The failure to grok is explicitly a failure to consolidate computation into specialized subgraphs.

### Weight Decomposition & SVD Spectrums

- **Pre-Grokking:** The singular value spectrum of the token embedding matrix is relatively flat for both conditions, meaning the representations are full-rank and uncompressed.
- **Post-Grokking (`pure`):** A sharp drop-off in the singular value spectrum emerges. The model has learned a low-rank embedding space (likely corresponding to the Fourier basis required for the modular arithmetic task).
- **Collapsed (`high_collapse`):** The spectrum remains flat, or degrades unpredictably. Label noise entirely prevents the embedding matrix from collapsing into the structured, low-rank subspace necessary for generalization.

## How to use this for future experiments

**Experiment A (Causal Circuit Rescue) integration:**
These tools lay the groundwork for a more refined version of circuit transplant. Instead of swapping entire matrices, future scripts can transplant only the top $k$ singular components (identified by `WeightDecomposition`) or specific attention heads (identified by `CircuitDiscoveryTool`).

**Running an analysis loop:**
```python
from src.analysis.circuit_analysis import CircuitDiscoveryTool, manual_transformer_forward

# Load model from checkpoint
tool = CircuitDiscoveryTool(model)
# x, y are from validation dataset
importance_scores = tool.compute_head_importance(x, y)
# High importance indicates the head is part of the core grokking circuit
```
