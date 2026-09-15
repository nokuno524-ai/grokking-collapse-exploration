# Transplant Success Matrix

Values are Mean ± Std Test Accuracy over seeds.

## Direction: pure->collapsed

| Component | low_collapse | medium_collapse | severe_collapse | high_collapse |
|---|---|---|---|---|
| baseline_base | 0.932 ± 0.000 | 0.859 ± 0.000 | 0.036 ± 0.000 | 0.348 ± 0.000 |
| baseline_donor | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| token_embed | 0.019 ± 0.000 | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.000 |
| self_attn_in_proj | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 |
| self_attn_out_proj | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 |
| linear1 | 0.021 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.000 |
| linear2 | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 |
| output_head | 0.048 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.015 ± 0.000 |

## Direction: collapsed->pure

| Component | low_collapse | medium_collapse | severe_collapse | high_collapse |
|---|---|---|---|---|
| baseline_base | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| baseline_donor | 0.932 ± 0.000 | 0.859 ± 0.000 | 0.036 ± 0.000 | 0.348 ± 0.000 |
| token_embed | 0.022 ± 0.000 | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.022 ± 0.000 |
| self_attn_in_proj | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.029 ± 0.000 | 0.019 ± 0.000 |
| self_attn_out_proj | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.000 |
| linear1 | 0.044 ± 0.000 | 0.039 ± 0.000 | 0.039 ± 0.000 | 0.039 ± 0.000 |
| linear2 | 0.025 ± 0.000 | 0.028 ± 0.000 | 0.030 ± 0.000 | 0.027 ± 0.000 |
| output_head | 0.037 ± 0.000 | 0.032 ± 0.000 | 0.019 ± 0.000 | 0.030 ± 0.000 |
