
### Transplant Success Matrix
Following the circuit transplant analysis, we performed a full multi-severity sweep (Pure vs. Low, Medium, Severe, High Collapse) for component feature attribution. Below is the transplant success matrix representing Mean ± Std zero-shot Test Accuracy across matched steps for various transplant directions (e.g. pure components transplanted into collapsed model base and vice-versa).

| Component | low_collapse | medium_collapse | severe_collapse | high_collapse |
|---|---|---|---|---|
| baseline_base (collapsed ckpt) | 0.932 ± 0.000 | 0.859 ± 0.000 | 0.036 ± 0.000 | 0.348 ± 0.000 |
| baseline_donor (pure ckpt) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| token_embed | 0.019 ± 0.000 | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.000 |
| self_attn_in_proj | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 |
| self_attn_out_proj | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 |
| linear1 | 0.021 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.000 |
| linear2 | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 |
| output_head | 0.048 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.015 ± 0.000 |

*Interpretation:* The data indicates that single-component swaps generally result in catastrophic accuracy loss (around ~1-4%) in zero-shot mode when substituting pure model components directly into collapsed base models (pure->collapsed). This signals that generalization recovery via single-component surgery is ineffective without subsequent retraining, highlighting that grokking circuits are distributed or highly co-dependent on adjacent layers and embedding geometry rather than being localized in isolated MLP layers or attention projections.
