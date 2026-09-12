## Robustness of headline claims

We stress-tested the grokking cliffs and weight-norm reduction claims across 27 defensible analytic regimes (varying smoothing windows, threshold metrics, and accuracy targets).

### Claim 1: Pure runs grok early (< 2000 steps), while Medium/Severe collapse never grok.
- **Survival Rate**: 3/27 (11.1%) regimes.
### Claim 2: Weight-norm reduction strictly correlates (negatively) with collapse severity.
- **Survival Rate**: 0/27 (0.0%) regimes (Benjamini-Hochberg corrected, alpha=0.05).

### Detailed Regime Breakdown

| Regime | Claim 1 Survives | Claim 2 Survives | Claim 2 Corr |
|--------|------------------|------------------|--------------|
| fixed_t0.8_w1 | ❌ | ❌ | 0.859 |
| fixed_t0.8_w10 | ❌ | ❌ | -0.380 |
| fixed_t0.8_w5 | ❌ | ❌ | 0.643 |
| fixed_t0.95_w1 | ✅ | ❌ | 0.859 |
| fixed_t0.95_w10 | ❌ | ❌ | -0.380 |
| fixed_t0.95_w5 | ✅ | ❌ | 0.643 |
| fixed_t0.9_w1 | ❌ | ❌ | 0.859 |
| fixed_t0.9_w10 | ❌ | ❌ | -0.380 |
| fixed_t0.9_w5 | ❌ | ❌ | 0.643 |
| max_2nd_der_t0.8_w1 | ❌ | ❌ | 0.859 |
| max_2nd_der_t0.8_w10 | ❌ | ❌ | -0.380 |
| max_2nd_der_t0.8_w5 | ❌ | ❌ | 0.643 |
| max_2nd_der_t0.95_w1 | ❌ | ❌ | 0.859 |
| max_2nd_der_t0.95_w10 | ❌ | ❌ | -0.380 |
| max_2nd_der_t0.95_w5 | ❌ | ❌ | 0.643 |
| max_2nd_der_t0.9_w1 | ❌ | ❌ | 0.859 |
| max_2nd_der_t0.9_w10 | ❌ | ❌ | -0.380 |
| max_2nd_der_t0.9_w5 | ❌ | ❌ | 0.643 |
| plateau_t0.8_w1 | ❌ | ❌ | 0.859 |
| plateau_t0.8_w10 | ❌ | ❌ | -0.380 |
| plateau_t0.8_w5 | ❌ | ❌ | 0.643 |
| plateau_t0.95_w1 | ✅ | ❌ | 0.859 |
| plateau_t0.95_w10 | ❌ | ❌ | -0.380 |
| plateau_t0.95_w5 | ❌ | ❌ | 0.643 |
| plateau_t0.9_w1 | ❌ | ❌ | 0.859 |
| plateau_t0.9_w10 | ❌ | ❌ | -0.380 |
| plateau_t0.9_w5 | ❌ | ❌ | 0.643 |