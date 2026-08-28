# Early-Warning Predictors for Grokking

Evaluation based on leave-one-severity-out validation.

## Predictor Rankings

| Predictor | Lead Time (steps) | Precision | Recall | FPR (on never-grok) |
|-----------|-------------------|-----------|--------|---------------------|
| loss_gap | 1875.0 | 0.118 | 0.133 | 1.000 |
| weight_norm_slope | 1396.7 | 0.500 | 1.000 | 1.000 |
| effective_rank | 0.0 | 0.000 | 0.000 | 0.000 |
| test_acc_curvature | 1396.7 | 0.500 | 1.000 | 1.000 |
| activation_sparsity | 0.0 | 0.000 | 0.000 | 0.000 |
| gradient_norm | 0.0 | 0.000 | 0.000 | 0.000 |

## Best Early-Warning Combo

**Recommended Combo**: `weight_norm_slope`
- Lead Time: 1396.7 steps
- Precision: 0.500
- Recall: 1.000
- FPR: 1.000

## Null-Hypothesis Check (Shuffled Labels)

To ensure predictors are better than chance, we shuffle the 'grokked' labels across runs.

| Predictor | Shuffled Precision | Shuffled Recall |
|-----------|--------------------|-----------------|
| loss_gap | 0.500 | 0.567 |
| weight_norm_slope | 0.500 | 1.000 |
| effective_rank | 0.000 | 0.000 |
| test_acc_curvature | 0.500 | 1.000 |
| activation_sparsity | 0.000 | 0.000 |
| gradient_norm | 0.000 | 0.000 |
