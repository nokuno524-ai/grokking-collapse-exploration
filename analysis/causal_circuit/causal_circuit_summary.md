# Causal circuit rescue — per-matrix rank trajectories

For each run we recorded the effective rank (`exp(-sum p log p)` over normalised singular values) of every 2D weight matrix at each saved checkpoint. The matrix with the largest single-step jump is the candidate 'circuit that grokking uses'.

| run | weight_decay | noise | seed | grokked | grok_step | top circuit | jump_step | rel_jump |
|---|---|---|---|---|---|---|---|---|
| wd0p3_noise0p0_seed42 | 0.3 | 0.0 | 42 | True | 3700 | `token_embed.weight` | 45000 | +0.323 |
| wd0p3_noise0p05_seed42 | 0.3 | 0.05 | 42 | True | 7300 | `transformer.layers.0.self_attn.in_proj_weight` | 10000 | -0.481 |
| wd0p3_noise0p15_seed42 | 0.3 | 0.15 | 42 | False | None | `transformer.layers.0.self_attn.in_proj_weight` | 10000 | -0.437 |
| wd1p0_noise0p0_seed42 | 1.0 | 0.0 | 42 | True | 1700 | `transformer.layers.0.linear2.weight` | 10000 | -0.359 |
| wd1p0_noise0p15_seed42 | 1.0 | 0.15 | 42 | False | None | `transformer.layers.0.linear2.weight` | 10000 | -0.278 |
| wd3p0_noise0p0_seed42 | 3.0 | 0.0 | 42 | False | None | `output_head.weight` | 10000 | +1.810 |

## Per-matrix top jump across all runs

Aggregated: average rel_jump per matrix, sorted descending.

| matrix | mean(rel_jump) | n_runs |
|---|---|---|
| `output_head.weight` | +0.272 | 6 |
| `token_embed.weight` | +0.245 | 6 |
| `transformer.layers.0.self_attn.in_proj_weight` | -0.233 | 6 |
| `transformer.layers.0.linear2.weight` | -0.167 | 6 |
| `transformer.layers.0.linear1.weight` | -0.129 | 6 |
| `transformer.layers.0.self_attn.out_proj.weight` | -0.120 | 6 |
| `pos_embed.weight` | -0.022 | 6 |
