# Curriculum Training: Data Mixing and Grokking Recovery

## Overview
This document explores the hypothesis: **Does starting training on collapsed data irreversibly poison circuits, preventing them from later grokking, or does it merely slow down an otherwise viable optimization path?** By utilizing dynamic mix proportions, we aim to measure exactly how resilient grokking mechanisms are under changing dataset compositions.

## Experimental Setup
We extended the pipeline to dynamically adjust `w(t)`, the proportion of collapsed data in a batch, across training using a time-varying scheduling function. We set up four schedule types:
1. **Constant**: Stable baseline for given static `w` combinations: 0.0, 0.25, 0.5, 0.75, 1.0.
2. **Linear**: Continuous linear transition from `w(0)` to `w(T)`.
3. **Cosine**: Soft smooth transition between boundary weights.
4. **Step**: Discrete boundary shift midway through training (e.g., pure to fully collapsed or vice versa at `T/2`).

All time-varying schedules are constrained to have a time-averaged `w` of roughly 0.5, allowing comparison to the steady state `constant_w0.5` configuration.

## Results Summary

*Pending Results Execution. We provide the structural layout of hypotheses given that analysis script results are determinable.*

### Does late exposure to pure data recover grokking?

**Hypothesis A (Irreversible Poisoning):** If early exposure to severely collapsed data creates local minima that the optimizer cannot escape—thereby permanently damaging representations or attention circuitry—grokking will fail regardless of whether clean data is eventually re-introduced.

**Hypothesis B (Optimization Slowness):** Alternatively, if the optimizer can eventually trace out the clean mathematical components from late exposure, late annealing of `w(t) -> 0` should recover grokking (perhaps at a later critical step or lesser final weight-norm peak).

### Does early exposure to pure data protect against later collapsed data?

Similarly, if early structure learned from purely synthetic/clean data is robust, transitioning to synthetic collapsed outputs (`w(t) -> 1`) may not disrupt already-forming attention circuits, providing protection that standard static mixing cannot.

## Mechanistic Follow-Up Opportunities
Based on the schedule comparison script outputs:
- **If Poisoning Dominates**: Investigate when (and which) attention heads lose alignment capability with positional representations. The point of no return can be located.
- **If Recovery Occurs**: Map the weight norm dynamics and use circuit transplantation (via `src/transplant_rescue.py`) to swap components at the point of recovery.
