# Comprehensive analysis: toy grokking + real contamination

Two experiments, two stress signals, two outcomes. The toy tests the **mechanism** (grokking) under label noise; the real experiment tests the **outcome** (LM degradation) under generative contamination. They are not point-paired but they should *agree* on the direction of effect.

## Toy summary

### Toy (exp_c_grid): 90 runs, wds=[0.3, 1.0, 3.0], noises=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

Mean final test_acc (grok_rate) per (wd, noise):

| wd \ noise | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|---|
| 0.3 | 1.000 (100%) | 0.979 (100%) | 0.961 (100%) | 0.804 (0%) | 0.698 (0%) | 0.291 (0%) |
| 1 | 0.984 (100%) | 0.977 (100%) | 0.914 (100%) | 0.842 (0%) | 0.758 (0%) | 0.236 (0%) |
| 3 | 0.147 (0%) | 0.121 (0%) | 0.078 (0%) | 0.104 (0%) | 0.077 (0%) | 0.053 (0%) |

Mean final Fourier concentration per (wd, noise):

| wd \ noise | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|---|
| 0.3 | 0.297 | 0.185 | 0.190 | 0.182 | 0.172 | 0.147 |
| 1 | 0.309 | 0.200 | 0.179 | 0.178 | 0.172 | 0.161 |
| 3 | 0.557 | 0.578 | 0.395 | 0.505 | 0.348 | 0.360 |

Mean final embedding effective rank per (wd, noise):

| wd \ noise | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|---|
| 0.3 | 31.77 | 40.27 | 40.43 | 40.49 | 39.91 | 39.66 |
| 1 | 25.71 | 33.98 | 36.72 | 36.88 | 41.61 | 38.81 |
| 3 | 9.80 | 9.83 | 12.55 | 11.49 | 12.96 | 13.17 |

## Real summary

### Real (contamination): 4 runs, ratios=[0, 10]

Mean final metrics per ratio:

| ratio_pct | n_seeds | perplexity | attn_effective_rank | repr_entropy | cos_sim_mean | distinct_2 | distinct_3 | distinct_4 |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 111.895 | 55.154 | 3.667 | 0.052 | 0.747 | 0.889 | 0.940 |
| 10 | 1 | 113.490 | 57.055 | 3.672 | 0.050 | 0.756 | 0.897 | 0.951 |

## Cross-experiment

### Cross-experiment: monotonicity of 'rank' under stress

- Toy: Spearman(noise, embedding_rank) = +0.295 (p=0.00471), n=90.
- Real: Spearman(ratio_pct, attn_effective_rank) = +0.775 (p=0.225), n=4 (only 2 ratios so far).

### Cross-experiment: weight-decay rescue

- Real: only weight_decay in {0.1} — no wd sweep yet, can't replicate the rescue. **TODO: rerun contamination at higher wd.**

## Reading guide

- The toy story: grokking dies above ~10% label noise. wd=1 partially rescues at 15% (higher accuracy, but no full grok). wd=3 kills grokking entirely. Fourier concentration tracks grokking; rank tracks the dimensionality the model uses to fit noise vs structure.
- The real story is incomplete. We need ratios beyond 0/10 and seed coverage. Once that lands, the equivalent **rank cliff** in attn_effective_rank should appear at the same place that perplexity blows up.
- The 'wd rescue' phenomenon in toy predicts that increasing weight decay during real LM training on contaminated data should also push the perplexity cliff to higher contamination. **Untested.**
