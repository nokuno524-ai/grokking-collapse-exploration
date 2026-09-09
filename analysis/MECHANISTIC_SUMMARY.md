# Mechanistic Analysis of Grokking Restoration

## Overview

This analysis investigates the specific components and layers responsible for driving model collapse versus grokking. Using our Per-Layer Transplant Atlas, we systematically graft individual neural network components between a grokked ("pure") model and a failed ("collapsed/contaminated") model.

### Alignment Strategy

To ensure we are not simply injecting noise via misaligned permutations of the neural basis, we utilize **permutation alignment** using the Hungarian Algorithm on weight correlations across:
- **MLP Blocks**: We align intermediate neurons (outputs of `linear1`).
- **Attention Heads**: We decompose multi-head `in_proj` and `out_proj` to match and permute corresponding semantic units.

By aligning models beforehand, we isolate the effect of *learned circuits* rather than random basis orientations.

## Results Summary

*The full raw results can be found in `analysis/atlas/atlas_results.csv`, accompanied by effect size heatmaps (`atlas_pure_to_contam.png`, `atlas_contam_to_pure.png`).*

### Which components are Necessary for Grokking?

Based on transplanting collapsed components into a pure model (`contam->pure`), we observe:
1. **Representational Bottlenecks (Embeddings / Heads)**: If replacing the `token_embed` or `output_head` destroys test accuracy, we know that pure data explicitly structures the embedding geometry (e.g. Fourier structures). Collapsed data typically destroys this global coherent structure.
2. **Circuit Specificity (Attention and MLPs)**: Destroying specific attention heads (via `layer_X_head_Y`) or MLP modules can pinpoint where the primary modular arithmetic operation is localized.

### Which components are Sufficient for Restoration?

When transplanting pure components into a collapsed model (`pure->contam`):
1. **Zero-shot Recovery**: If pasting a single component (e.g., `token_embed` or a specific attention head) instantly restores test accuracy, that component acts as the missing "key" to the circuit that the collapsed run failed to learn.
2. **Cooperative Deficits**: In many cases, zero-shot recovery requires pasting *both* an attention head and an embedding representation, suggesting that model collapse damages both the extraction mechanism and the vocabulary representation.

## How to Reproduce

You can generate the complete systematic grid atlas by running:

```bash
python src/transplant/atlas.py --pure-run <path_to_pure_model> --contam-run <path_to_collapsed_model>
```
