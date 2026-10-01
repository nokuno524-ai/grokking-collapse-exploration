# Mechanistic Interpretability Summary

This document summarizes the mechanistic interventions and findings, specifically related to the statistical methods and systematic cross-run transplant procedures recently introduced.

## Statistical Rigor Protocol

To avoid over-interpreting noise in small neural networks (especially in failure modes where models may bounce stochastically near random chance), we enforce rigorous statistical standards:
- **Bootstrap Confidence Intervals**: All performance deltas computed via cross-run transplants must be reported with 95% bootstrap confidence intervals. The `src.rigorous_stats.stats_module` handles this with `bootstrap_ci`.
- **Effect Sizes**: The accuracy distributions across validation batches are recorded to capture variation within a checkpoint evaluation.
- **Reporting**: All metric logging formats use `n`, `mean`, and `95% CI` implicitly through `format_stat_result`.

## Cross-Run Component Transplants

To determine if a representation learned by a pure (grokked) seed generalizes or is isomorphic across seeds, we swap state dict components systematically:
- We support full module patching (`patch_state_dict`) as well as surgical single-head patching (`patch_head_state_dict`), located in `src/transplant/patching.py`.
- **Induction Training**: If a zero-shot transplant doesn't work out-of-the-box due to minor misalignment, we optionally freeze the entire network except the transplanted component and fine-tune it via `induction_train`.
- Outputs from `run_transplant_sweep` create a full (donor_seed x recipient_component) success matrix.

## Ablation Atlas

A targeted ablation suite (`src/transplant/atlas.py`) allows:
- `zero_ablate_component`: Destroys the component.
- `mean_ablate_component`: Replaces activations with the empirical mean, preserving the baseline but destroying signal.
- `resample_ablate_component`: Randomly permutes weights, breaking spatial structure while retaining exact weight distribution spectra.

## Reproducibility

Unit tests for slicing logic and ablation properties are handled via `tests/test_transplant.py` and `tests/test_atlas.py`, which utilize tiny deterministic transformers to ensure CPU-fast CI. Execute via `PYTHONPATH=. uv run --with pytest pytest tests/`.
