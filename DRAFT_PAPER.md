# Grokking Cliffs: The Causal Mechanisms of Model Collapse

## Abstract
Recent work has demonstrated that models trained on recursively generated synthetic data experience "model collapse," a phenomenon where the generative distribution narrows and rare events are lost. Concurrently, the phenomenon of "grokking"—delayed generalization occurring long after overfitting—has been identified as a window into the mechanistic formation of robust internal circuits. In this paper, we bridge these two fields by investigating how varying degrees of distributional collapse in training data disrupt the mechanistic circuit-formation process necessary for grokking. We analyze the causal roles of attention heads and track the geometric evolution of the weight matrices to understand the precise failure modes induced by data contamination. Our controlled study reveals a sharp grokking cliff as data quality deteriorates, providing a novel perspective on how model collapse fundamentally alters the learning trajectory.

## Introduction
The rapid adoption of Large Language Models (LLMs) has led to an increasing fraction of the internet being composed of synthetic text. Training future models on this data is known to induce "model collapse" (Shumailov et al., 2024). While the macroscopic effects—loss of diversity and factual degradation—are well documented, the underlying mechanistic changes in the model's learning dynamics remain opaque. We study this through the lens of "grokking" (Power et al., 2022), training a small transformer on modular arithmetic. By subjecting the model to varying levels of synthetic data contamination, we characterize the causal breakdown of the generalization process.

## Related Work
* **Grokking**: Generalization beyond overfitting on small algorithmic datasets, characterized by delayed phase transitions and the eventual formation of sparse, interpretable circuits (Power et al., 2022; Nanda et al., 2023).
* **Model Collapse**: The degradation of models trained on generated data, leading to a loss of the original distribution's tails (Shumailov et al., 2024; Dohmatob et al., 2024).
* **Mechanistic Interpretability**: Understanding the internal algorithms of neural networks. Recent techniques include Causal Head Gating to isolate specific circuit components and weight matrix decomposition (SVD) to track rank evolution.

## Method
We train a 1-layer, 4-head Transformer on the task `(a + b) mod 59` using a 30% training fraction. We simulate model collapse by replacing a fraction of the training targets with samples from a temperature-warped distribution that favors common results. We investigate three conditions: pure data (0% collapse), medium collapse (15%), and severe collapse (50%).

Our mechanistic investigation employs two primary techniques:
1. **Causal Head Gating**: We inject learnable scalar gates on the output of each attention head and train them with an $L_1$ penalty while freezing the base model. This allows us to continuously classify heads as facilitating, irrelevant, or interfering across the training trajectory.
2. **Weight Forensics**: We track the $L_2$ distance of the weights from initialization, the effective rank (via SVD Shannon entropy) of key matrices, approximate the top Hessian eigenvalue during training, and compute a neuron importance participation ratio based on activation magnitude and gradient.
3. **Data Quality Metrics**: We quantify the distributional shift using KL divergence and Shannon entropy. Furthermore, we compute the memorizability score of the dataset to predict which samples encourage memorization.

## Experiments
We train the model across 5 random seeds for each collapse condition. We periodically log test accuracy, training loss, and the Fourier spectrum concentration of the token embeddings. After training, we perform our causal gating and weight forensics analyses on the saved checkpoints.

## Results
1. **The Grokking Cliff**: We observe a sharp transition where models grok perfectly on pure data and low collapse (5%) but fail completely at 15% medium collapse and beyond. This is visualized in Figure 1 (`main_grokking_curves.pdf`).
2. **Data Shift**: As the collapse level increases, the Shannon entropy of the target distribution decreases monotonically, while the KL divergence from the uniform distribution increases sharply (Figure 4: `data_quality.pdf`).
3. **Causal Head Roles**: The head gating analysis reveals that under pure conditions, specific attention heads solidify their facilitating roles exactly as the model transitions from memorization to generalization. Under severe collapse, these roles fail to consolidate or become erratic (Figure 2: `head_roles_heatmap.pdf`).
4. **Weight Geometric Evolution**: The effective rank of the token embeddings drops significantly during the grokking phase in the pure condition, indicating circuit compression. In the collapsed conditions, this rank remains artificially high or follows an altered trajectory, suggesting that the model remains stuck in a high-dimensional memorization regime (Figure 3: `weight_rank_evolution.pdf`).

## Discussion
Our findings demonstrate that model collapse is not merely a statistical artifact but a causal disruptor of circuit formation. The contaminated data creates an optimization landscape where the transition from high-rank memorization circuits to low-rank generalizable circuits is no longer favorable. This highlights the fragility of the grokking phenomenon to even minor (15%) systemic label corruptions, challenging the robustness of future models trained on synthetic data.

## References
* Power, A., et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.
* Nanda, N., et al. (2023). Progress Measures for Grokking via Mechanistic Interpretability.
* Shumailov, I., et al. (2024). The Curse of Recursion: Training on Generated Data Makes Models Forget.
* Dohmatob, E., et al. (2024). A Tale of Tails: Model Collapse as a Change in Scaling Laws.
