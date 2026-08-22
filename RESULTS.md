# Data-Driven Project Findings: Grokking and Model Collapse

This document is generated automatically from the centralized experiment registry (`results/registry.json`), analyzing data across 200+ runs focusing on the impact of label noise (model collapse) on generalization.

## 1. Collapse Prevents Grokking (Headline Finding)

Training on collapsed data directly impacts the model's ability to grok.

- **Pure Data Grokking Rate:** 100% (n=7)
- **Severe Collapse Grokking Rate:** 0% (n=7)
- **Statistical Significance:** p = 0.0005 (Permutation test, 10,000 permutations)

The permutation test confirms that the reduction in grokking incidence under severe collapse conditions is statistically significant, validating the core hypothesis that recycled data impairs delayed generalization.

## 2. Effect Sizes by Collapse Severity

Comparing final metrics across severity conditions (mean [95% CI]):

| Condition | Final Test Accuracy | Final Weight Norm | Final Embedding Rank |
|---|---|---|---|
| Pure | 0.988 [0.965, 1.000] | 32.2 [30.2, 35.1] | 25.8 [24.5, 27.6] |
| Low Collapse | 0.973 [0.957, 0.986] | 35.9 [35.1, 36.5] | 33.4 [32.3, 34.6] |
| Medium Collapse | 0.853 [0.837, 0.869] | 41.2 [38.4, 43.8] | 37.1 [36.5, 37.7] |
| High Collapse | 0.285 [0.248, 0.330] | 56.4 [53.9, 59.5] | 37.9 [36.9, 38.8] |
| Severe Collapse | 0.039 [0.035, 0.044] | 59.4 [56.9, 62.1] | 37.3 [36.6, 38.0] |

Weight-norm reduction correlates strongly with collapse severity, providing a continuous metric of degradation.

## 3. Caveats and Open Questions

- **Scale Limitation:** The primary findings are based on a 1-layer transformer (214K params). While indicative of fundamental dynamics, scaling up to larger architectures may introduce nuanced behaviors.
- **Real vs. Synthetic:** The observed equivalence between random label noise and temperature-warped collapse warrants further investigation on natural language tasks to confirm generalization.
- **Weight Decay Interaction:** Weight decay modulates the grokking cliff threshold. The precise interaction between regularizers and contamination ratios remains an open theoretical question to be fully characterized.
