# Mechanistic Interpretability: Grokking vs. Collapse

This document summarizes the circuit-level findings and their implications for understanding the interplay between grokking and model collapse at the mechanistic level.

## 1. Circuit Discovery & Activation Patching

Our circuit discovery analysis focuses on the causal importance of individual attention heads in learning the modular addition task.
By applying zero and mean ablation strategies to the `out_proj.weight` matrix of specific heads within the `ModularArithmeticTransformer`, we can isolate each head's contribution to overall performance.

Comparing the causal importance pre-grokking and post-grokking reveals a **Grokking Circuit**:
- **Pre-Grokking (Memorization Phase):** Causal importance is distributed widely across multiple heads. The model relies on memorizing specific combinations rather than abstract rules.
- **Post-Grokking (Generalization Phase):** Causal importance concentrates significantly into a small subset of "grokking heads." The transition point is sharp. These key heads implement the abstract modular arithmetic logic (likely tied to the Fourier features observed in earlier analyses).

When the model is exposed to data contaminated by **Model Collapse (label noise ≥ 15%)**, this circuit formation is disrupted. The attention heads remain in a distributed, memorization-heavy state, unable to form the sparse, generalized subnetwork required for grokking.

## 2. Attention Head Taxonomy

Using standard interpretability methods, we classified the attention heads into functional categories:

*   **Previous-Token Heads:** Primarily attend to the token at position `i-1`. In the context of our 2-token `(a, b)` input, this allows the second token to access the first token's value.
*   **Duplicate-Token Heads / Induction Heads:** While more relevant for longer sequences (like in autoregressive language models), we track these to understand if standard induction circuits attempt to form during training.

**Mechanistic Implication of Collapse:** Under severe collapse conditions (high label noise), the attention patterns fail to stabilize into discrete, specialized roles. The taxonomy heatmap shows smeared, uncertain attention distributions. The noisy labels prevent the gradient descent process from successfully separating and specializing the attention heads for rule-based generalization.

## 3. Weight-Level Analysis (Phase Transitions)

To understand *how* the circuit changes during grokking, we analyzed the individual weight matrices across checkpoints:

1.  **Effective Rank (SVD Entropy):** During the initial memorization phase, the effective rank of the weight matrices remains relatively high. At the grokking transition point, we observe a rapid drop in effective rank (compression). This correlates perfectly with the emergence of the Grokking Circuit and the increase in Fourier concentration.
2.  **Cosine Similarity:** By tracking the cosine similarity of weight vectors between consecutive training steps, we can mechanically pinpoint the phase transition. During grokking, the cosine similarity momentarily drops, indicating a massive reorganization of weights (rapid movement in parameter space) as the model shifts from the memorization minima to the generalization minima.

**Why Collapse Prevents Grokking:**
In conditions of medium to severe collapse, this phase transition (the rapid drop in cosine similarity and subsequent drop in effective rank) *never occurs*. The gradient noise induced by the corrupted data creates a highly irregular loss landscape. The optimizer (even with strong weight decay) gets trapped in local memorization minima and cannot find the narrow path to the compressed, generalized solution. Weight norm reduction correlates with collapse severity because the model cannot confidently update its weights towards a generalized solution, leading weight decay to aggressively push the unresolved weights towards zero.
