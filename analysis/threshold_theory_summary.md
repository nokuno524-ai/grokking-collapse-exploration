# Threshold Theory — Empirical Fit (Experiment C)

## Theoretical prediction

From the cleanup-phase balance condition (★) in `src/threshold_theory.py`:

$$\eta^*(\lambda, p, d)  \;\approx\;  C \cdot \lambda \cdot \sqrt{p \cdot d_{\text{model}}}$$

Holding p and d fixed (this study: p=59, d=128), the prediction reduces to **η* ∝ λ¹** (i.e. exponent b = 1 in η* = C·λ^b), valid only in the regime where (★★) holds — i.e. weight decay does not by itself destabilize the memorization solution.

## Setup

- Grid: wd ∈ [0.3, 1.0, 3.0], noise ∈ [0.0, 0.05, 0.1, 0.15, 0.2, 0.3], n_seeds = 5
- η*(λ, seed) := smallest noise level where the run failed to reach test_acc ≥ 0.95.
- If all noise levels grokked → right-censored at max(noise) + 0.01.
- If no noise level grokked (regime II / decay too large) → η* = 0 and excluded from the fit.

## Regime II detected at wd ∈ [3.0]

These wd values prevent grokking even at noise=0. Per the derivation, this corresponds to violation of the memorization-stability condition (★★): λ·||θ_mem|| > ||∇L_clean(θ_mem)||. These points are excluded from the cliff-shift fit because there is no cliff to fit — the model fails everywhere.

## Per-(wd, seed) cliff position

| wd | seed | η* |
|---|---|---|
| 0.3 | 42 | 0.1500 |
| 0.3 | 43 | 0.1500 |
| 0.3 | 44 | 0.1000 |
| 0.3 | 45 | 0.1500 |
| 0.3 | 46 | 0.1500 |
| 1.0 | 42 | 0.1000 |
| 1.0 | 43 | 0.1000 |
| 1.0 | 44 | 0.0000 |
| 1.0 | 45 | 0.1000 |
| 1.0 | 46 | 0.1000 |
| 3.0 | 42 | 0.0000 |
| 3.0 | 43 | 0.0000 |
| 3.0 | 44 | 0.0000 |
| 3.0 | 45 | 0.0000 |
| 3.0 | 46 | 0.0000 |

## Power-law fit

- Fitted η* = **0.1000 · λ^-0.269**
- R² = 0.640
- Bootstrap 95% CI on b: [-0.337, -0.000]
- Theory predicts b = 1.0; empirical b = -0.269.

**Verdict:** predicted b=1 is *outside* the 95% CI [-0.337, -0.000]. Either the constant assumptions in the derivation are wrong (e.g. ||∇L_noise|| also scales with λ via Adam preconditioning), or the discretisation of η in the grid is too coarse to resolve the cliff shift. Recommend: rerun a finer noise sweep at η ∈ {0.06, 0.08, 0.10, 0.12, 0.14, 0.16} for two wd values.

## How to reproduce

```bash
python src/threshold_theory.py --grid-dir results/exp_c_grid --output-dir analysis/
```
