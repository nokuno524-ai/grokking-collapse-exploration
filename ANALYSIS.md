# Analysis Tools

This document outlines the analysis tools available in this repository to study grokking and model collapse. These tools extract mathematical and mechanistic insights from training checkpoints to explain exactly *when* and *how* the model shifts from memorizing to generalizing (or failing to).

## 1. Attention Pattern Evolution (`analysis/attention_evolution.py`)
This module tracks the behavior of the `ModularArithmeticTransformer`'s attention heads across training:
- Computes **Attention Entropy** (using Shannon entropy) to measure if a head distributes its attention broadly or focuses on specific tokens.
- Generates **Attention Heatmaps** across layers/heads over time to trace the circuit formation visually.
- Helps identify specific heads specializing in certain operations before the grokking transition.

**Usage:**
```python
from analysis.attention_evolution import analyze_attention_evolution, plot_attention_entropy

metrics = analyze_attention_evolution("results/pure", model_config, sample_inputs)
plot_attention_entropy(metrics, "attention_entropy.png")
```

## 2. Weight Space Analysis (`analysis/weight_space.py`)
This module analyzes the evolution of weight norms, structures, and cross-checkpoint comparisons:
- Computes **Effective Rank** (Participation Ratio) using Shannon entropy applied to the singular values of each weight matrix.
- Evaluates the **L2 Norms** over time per layer.
- Computes the **Cosine Distance** between checkpoints to trace weight divergence.
- Includes **Activation CKA** (Centered Kernel Alignment) to compare the similarity of network features between pure and collapsed models.

**Usage:**
```python
from analysis.weight_space import analyze_weight_space

metrics = analyze_weight_space("results/pure")
# Returns a dictionary with layer norms, effective ranks, and cosine distances.
```

## 3. Grokking Detector (`analysis/grokking_detector.py`)
This script implements a mathematical approach to identifying the grokking transition:
- Extracts the accuracy curve from `results.json` histories.
- Uses `scipy.signal.savgol_filter` to smooth test accuracy and compute its first and second derivatives.
- Detects the **Grokking Point** as the step where the second derivative (acceleration) peaks, marking the start of the sudden rise in test accuracy.
- Computes the **Grokking Gap** and returns distinct training phase boundaries: Memorization, Grokking, and Generalization.

**Usage:**
```python
from analysis.grokking_detector import analyze_training_run

grokking_info = analyze_training_run(results_history)
print(grokking_info['phases'])
```
