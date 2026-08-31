# Circuit Transplant Cross-Seed Replication Findings

This document summarizes the findings from the circuit-transplant cross-seed replication experiments, which build on our initial single-seed experiments to ensure statistical rigor in determining whether model collapse universally corrupts specific components (like a specific attention head or MLP) or leads to a global degradation that cannot be rescued with simple component swapping.

## Methodology

We performed a surgical component transplant across 5 different initialization seeds for each condition (pure vs. pure, pure vs. low collapse, pure vs. medium collapse, pure vs. severe collapse). For each seed, a donor component from the pure model was pasted into the contaminated run's matched checkpoint.

We computed effect sizes (Cohen's d, paired) and 95% bootstrap confidence intervals for the test accuracy difference (zero-shot) before and after transplantation. A crucial metric is whether the transplant effect replicates across seeds — meaning the effect consistently maintains the same sign.

## Findings

*(Note: The findings will be populated by running `src/transplant/replication_harness.py`. Please check `analysis/transplant_replication/replication_summary.md` and the forest plots for the generated output data once the full GPU cluster run completes).*

Based on the statistical aggregation:
- Components that consistently rescue test accuracy when transplanted indicate that those components carry the grokking mechanisms, and their corruption under contamination is the localized reason for failure to grok.
- A lack of consistent replication (e.g., positive effect in one seed, negative in another) suggests that either the component relies heavily on the specific initialization or that model collapse fundamentally leads to global degradation (diffuse corruption), rendering single-component transplants insufficient for rescue.

## Usage

To regenerate these results, you can use the replication harness:
```bash
python -m src.transplant.replication_harness
```

And to plot the forest plots:
```bash
python -m src.transplant.plot_replication
```
