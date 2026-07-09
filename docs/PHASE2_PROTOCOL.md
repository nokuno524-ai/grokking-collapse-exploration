# Phase 2 Protocol: Collapse vs. Noise Mechanistic Experiments

## 1. Goal & Hypothesis

The central open question from Phase 1 is whether "model collapse" is structurally distinct from uniform label noise in preventing grokking. Phase 1 test accuracy data showed that a matched noise rate was statistically indistinguishable from a model collapse condition.

**Hypothesis**: If model collapse is merely "effective label noise," mechanistic metrics (grokking onset, Fourier concentration dynamics, and weight matrix rank) will be identical across matched noise and collapse conditions. If it is structurally different, these metrics will differ, even if final test accuracy is similar.

## 2. Experimental Design

Three separate analyses will be conducted across `pure`, `collapse`, and `noise` matched conditions using 1-layer Transformers on modular arithmetic `(a+b) mod p`.

### 2.1. `phase2_collapse_vs_noise.py`: The Collapse vs Noise Base Comparison
- Calculates empirical KL divergence and "effective information content" (bits preserved from max entropy) of corrupted distributions.
- Tracks `grokking_step` onset and `weight_norm` evolution during training under matched noise rates.
- *Prediction*: If they differ, collapse may preserve "easy" sub-patterns (less information lost) than random noise, leading to differences in onset timing.

### 2.2. `phase2_fourier_analysis.py`: Fourier Concentration Dynamics
- Fourier basis coefficients are extracted directly from the embedding matrices at each training step using energy (`.abs() ** 2`).
- Fourier concentration metrics are stored dynamically to produce evolutionary heatmaps across training steps.
- *Prediction*: Pure data will cleanly concentrate on the top-k modes. If collapse is structurally different from noise, it may show "false concentration" on incorrect or distinct modes, whereas noise might cleanly dilute all non-signal modes equally.

### 2.3. `phase2_weight_analysis.py`: Deep Weight Rank / Circuit Formation
- Applies SVD to each network matrix (`token_embed`, `pos_embed`, `attn_out`, `ff1`, `ff2`, `output_head`).
- Computes effective rank based on the Shannon entropy of singular values $H = \exp(-\sum s_i \log(s_i))$ where $s_i$ are normalized.
- *Prediction*: Identifies the precise moment "circuits" form (sharp rank reduction). Collapse might show different rank reduction timing or magnitude vs uniform noise, signifying an structural over-fitting rather than simple disruption.

## 3. Execution

- Environment: `uv` and Python.
- Output: `results/phase2_collapse_vs_noise/`, `results/phase2_fourier_analysis/`, `results/phase2_weight_analysis/`.
- Summary generated via JSON logging on a per-seed basis for aggregation and CI bounds.
