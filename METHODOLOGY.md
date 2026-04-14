# Methodology

This document outlines the methodology for evaluating the intersection of large language model grokking behavior and distributional collapse.

## Metrics Used

To understand the progress and outcome of the training experiments, we use a combination of metrics tailored to analyze model generalization and collapse:

1. **Mode Collapse Score:**
   Measures the degree to which the model's predictions focus entirely on a small number of modes rather than representing a wider diversity. The score is scaled between `0.0` (predictions perfectly uniform across all classes) and `1.0` (all predictions are exactly the same class). It is calculated using information entropy: $1.0 - \frac{\text{Entropy}(\text{predictions})}{\log(\text{num\_classes})}$.

2. **KL Divergence Shift:**
   Calculates the Kullback-Leibler (KL) divergence between the marginal distribution of model predictions and the true target distribution. This identifies if the model's output distribution is fundamentally mismatched with the truth. High KL divergence indicates a significant distribution shift.

3. **Loss of Complexity:**
   Represented by the effective rank of the model embeddings. As grokking takes hold, a model frequently distills a simpler, robust internal representation of algorithms like modular arithmetic (often manifesting as a recognizable Fourier spectrum structure). On the other hand, in a collapsed state, the model may also have low effective rank but without generalizing, simply lacking complex representation entirely.

4. **Memorization Score:**
   Calculates the gap between training accuracy and test accuracy. High values (near `1.0`) imply the model has simply memorized the corrupted dataset and fails to generalize to the clean test set.

5. **Fourier Concentration:**
   Used as a specific heuristic for modular arithmetic tasks to evaluate if the model is learning the true mathematical structure (Fourier components) underlying the problem. High Fourier concentration tracks with the discrete Fourier circuit forming during late-stage grokking.

## Phase Diagram Interpretation

To understand the dynamic interplay of collapse factors and optimization, a parameter sweeping tool is used to generate a 2-dimensional phase diagram. For instance, the collapse level (synthetic data ratio) can be swept on the x-axis, and learning rate or weight decay on the y-axis.

- **Grokking Region:** Indicates regions where the model cleanly achieves >95% test accuracy.
- **Memorization Region:** Indicates regions where the model fits the training data perfectly (>90% train accuracy) but fails to generalize (<90% test accuracy).
- **Collapse / Failed Region:** Indicates regions where the model utterly fails to learn the structure, manifesting low train and test accuracies.

By varying hyperparameters, we can identify "tipping points" where distributional collapse causes training runs to suddenly transition from grokking to mere memorization or outright failure.
