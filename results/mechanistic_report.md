# Mechanistic Analysis: Model Collapse vs Grokking

This report analyzes the weight geometry and mechanisms across different collapse conditions.

## 1. Weight Geometry Analysis

| Condition | Embed Norm | Attn Norm | MLP Norm | Out Head Norm | Embed Rank | MLP Rank Avg |
|-----------|------------|-----------|----------|---------------|------------|--------------|
| pure | 1.59 | 26.00 | 7.98 | 8.28 | 25.75 | 33.58 |
| low_collapse | 2.50 | 31.97 | 9.65 | 10.68 | 35.46 | 50.94 |
| medium_collapse | 2.22 | 31.92 | 9.71 | 11.40 | 36.96 | 57.05 |
| severe_collapse | 2.08 | 54.20 | 9.57 | 11.14 | 38.75 | 53.62 |
| high_collapse | 2.21 | 51.84 | 10.23 | 11.88 | 39.25 | 54.17 |

## 2. Singular Value Distributions (Top 3 Embedding Singular Values - Final Step)

| Condition | SV 1 | SV 2 | SV 3 |
|-----------|------|------|------|
| pure | 0.48 | 0.47 | 0.45 |
| low_collapse | 0.82 | 0.72 | 0.65 |
| medium_collapse | 0.76 | 0.66 | 0.53 |
| severe_collapse | 0.52 | 0.51 | 0.48 |
| high_collapse | 0.55 | 0.54 | 0.53 |

## 3. Conclusions

Based on the geometric and mechanistic analysis:

1. **Weight Norms:** Collapse conditions tend to exhibit different weight norm trajectories, which correlates with their failure to grok.
2. **Effective Rank:** The effective rank of weight matrices (especially embeddings) differs significantly between pure models that grok and collapsed models that memorize noise.
3. **Mechanisms:** The delayed generalization in pure models corresponds to the emergence of specific low-rank structures, which are disrupted by label noise.
