# Results Summary: Model Collapse and Grokking Interplay

This document synthesizes findings on how model collapse (training on synthetic or contaminated data) influences the generalization behavior known as *grokking* in a simple transformer trained on modular arithmetic.

## Key Findings

1. **Grokking is Inhibited by Collapse:**
   In pure and low-collapse conditions, models reliably undergo delayed generalization (grokking) where test accuracy suddenly spikes long after training loss plateaus. As the collapse severity increases (medium, severe, high), the grokking threshold is delayed or prevented entirely.

2. **J-Lens Analysis:**
   Using our simplified Jacobian lens (`src/analysis/jlens.py`), we track how explicitly intermediate representations (embeddings, transformer output, layer norm) project into the vocabulary space.
   - **Dimensionality:** Pure models maintain higher dimensional and coherent semantic spaces, whereas severe collapse drastically reduces the rank of intermediate representations.
   - **Entropy:** Vocabulary-projected entropy drops earlier in pure runs as the network confidently groks, while collapsed models linger with diffuse, high-entropy projections.

3. **Feature Evolution (Fourier & Attention):**
   - **Fourier Concentration:** Pure datasets foster a rapid increase in Fourier spectrum peak magnitude (`results/feature_evolution.png`), signaling the formation of algebraic circuits. Highly collapsed datasets fail to develop these strong frequency components.
   - **Attention Specialization:** The entropy of attention heads (`src/analysis/feature_evolution.py`) decreases sharply during grokking for clean data. Under severe collapse, attention entropy remains elevated, meaning the heads fail to specialize into rigid, low-entropy operations.

4. **Attention Patterns Visualization:**
   Heatmaps (`src/viz/attention/attention_patterns_step_*.png`) show that:
   - Early steps (`10000`): Attention is uniformly distributed across positions 0 and 1 for all conditions.
   - Mid steps (`30000`): Pure and low collapse start showing distinct, sharp attention maps (e.g. self-attention focused entirely on specific positions or diagonal structures).
   - Late steps (`50000`): Collapsed models fail to converge on sharp attention structures, maintaining fuzzy distributions.

## Summary Statistics

| Condition | Grokking Outcome | Fourier Pattern | Attention Specialization | J-Space Rank |
| --- | --- | --- | --- | --- |
| Pure | Swift | Sharp peaks | High (low entropy) | High |
| Low Collapse | Delayed | Muted peaks | Moderate | Medium |
| Medium Collapse| None | Weak/No peaks | Low (high entropy) | Low |
| Severe Collapse| None | Noise | None | Very Low |
| High Collapse | None | Noise | None | Very Low |

## Conclusion
Model collapse effectively squashes the necessary circuit formation (Fourier frequencies, sharp attention, high-rank intermediate spaces) required for a model to grok an algorithmic task. The degraded synthetic data lacks the necessary high-fidelity signal to cross the threshold into robust generalization.