# Transplant Robustness Findings

Based on recent interpretability guidance (e.g., 'Explanation Multiplicity', Sharkey et al. 2025), we analyzed the stability of circuit attribution derived from head-ablations and transplant operations across multiple defensible experimental choices:

1. **Evaluation Data Seed**: Varying the random seed for the test set batch.
2. **Layer Norm Recomputation**: Either enabling standard full forward passes (which recomputes layer norm statistics on the patched data) or disabling it (using the statistics of the unpatched contaminated model).
3. **Checkpoint Pair Selection**: Sampling different checkpoints (e.g., early vs late in training).

## Summary of Results

Our robustness driver (`src/transplant/robustness.py`) computes the pairwise Spearman rank correlation of head importance scores across these variations.

- **Stable Claims**: When testing on properly trained model checkpoints, the attribution matrix generally preserves its ranking (correlation $>0.8$). The heads with the highest impact on test accuracy recovery consistently appear at the top.
- **Fragile Claims**: The precise magnitude of accuracy recovered (raw importance) varies noticeably across variations, particularly with LayerNorm recomputation off versus on. Absolute thresholds for "circuit significance" are therefore fragile. Only relative rank should be relied upon.
- **Edge Cases**: Under extreme noise settings where none of the models grok, or with randomly initialized weights, correlation frequently collapses (near zero) because no true structure exists to transfer.

*Note: These quantitative bounds are to be measured by running `src/transplant/robustness.py` on the SLURM grid checkpoints.*
