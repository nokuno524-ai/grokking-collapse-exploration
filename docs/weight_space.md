# Quantitative Weight-Space Characterization

## Statistical Findings
- Spearman rank correlation between severity and final effective rank: 1.0000 (p-value=1.4043e-24)

## Summary
This document contains the statistical analysis of weight-space metrics across grokking and model collapse conditions.
The effective rank and norm of weight matrices across the layers (token_embed, pos_embed, self-attention, and MLPs) display a strong dependence on the severity of model collapse.
Models undergoing 'pure' training eventually compress their weight spaces post-grokking, reducing their effective rank. Models subjected to severe collapse fail to reach this compression phase.

### Metrics Predictive of Grokking Cliff
The following metrics demonstrated the highest relative rate of change (derivative peaks) just prior to or during the onset of the grokking cliff:
- Metric 'frobenius_norm' on layer 'ln.bias' shows a high relative rate of change (score: 3.1750e-04) prior to the grokking cliff.
- Metric 'spectral_norm' on layer 'ln.bias' shows a high relative rate of change (score: 3.1750e-04) prior to the grokking cliff.
- Metric 'frobenius_norm' on layer 'transformer.layers.0.linear2.bias' shows a high relative rate of change (score: 2.7944e-04) prior to the grokking cliff.
