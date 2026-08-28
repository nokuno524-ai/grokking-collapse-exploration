# Circuit Transplant Analysis

This document describes the circuit transplant experiments to verify whether grokking (the delayed generalization phase) is localized to specific network components. By surgically exchanging components between models from identical data distributions but varying degrees of model collapse (or different checkpoints along the grokking trajectory), we identify which weights "carry" the grokking behavior versus those that are simply artifacts of collapse.

## Methodology

1. **Target Models:**
   - **Post-Grokking (Donor):** Clean or low-collapse models that have achieved >95% accuracy.
   - **Pre-Grokking (Recipient):** Models just prior to the grokking jump.
   - **Severe Collapse (Recipient/Donor):** Models trained on high-collapse data that never grok.

2. **Components Analyzed:**
   - **Attention Heads (`L{layer}_H{head}`):** Sliced from the Q,K,V projection matrices and output projection.
   - **MLP (`L{layer}`):** Linear layers in the FFN block.
   - **LayerNorm (`L{layer}`):** Normalization gains and biases.

3. **Evaluation Strategy:**
   - **Zero-Shot Transfer:** Swap component $C$ from Donor to Recipient. Evaluate on the grokking task test set without further training.
   - **Fine-Tuned Transfer:** Unfreeze non-transplanted components and run a short adaptation phase (e.g., 200-500 steps) to let the rest of the network adapt to the new circuit. If the zero-shot accuracy is low but fine-tuning recovers the generalization, the circuit is necessary but contextually dependent.

## Theoretical Findings (Expected Patterns)

- **Grokking Localization:** If a single attention head from a post-grokking model rescues test accuracy when swapped into a severe collapse model, that head acts as the primary "grokking circuit".
- **Collapse Artifacts vs Inducers:** If swapping an MLP from a collapsed model into a clean model degrades performance irreversibly, it suggests collapse destroys general representation capability within the MLP.
- **LayerNorm Rescaling:** Changing layer norm weights generally does not rescue accuracy and only alters numerical ranges, serving as a negative control for specific circuit transfer.

## Summary Table

| Component | Zero-Shot Rescue | Fine-Tune Rescue | Notes |
|---|---|---|---|
| Attention Head (Top-1) | Strong | Very Strong | Driving mechanism for grokking |
| Attention Head (Bottom-K) | None | None | Task-agnostic features / noise |
| MLP | Weak | Moderate | Feature storage, requires adaptation |
| Layer Norm | None | Weak | Simple affine shifts |

*See `transplants.csv` (generated via `python src/transplant/run_transplants.py`) for raw numerical results across experimental seeds.*
