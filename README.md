# Grokking Cliffs: Label-Noise Rate, Weight Decay, and the Scarcity Dissociation

A controlled study of how label-noise rate, weight-decay strength, and training-data scarcity each modulate **grokking** (delayed generalization) on the modular-arithmetic task `(a + b) mod p`.

## What the data show

After 230 toy runs (1-layer transformer, 214K params, p=59) and 4 GPT-2-medium real-LM runs:

1. **Sharp grokking cliff in label-noise rate.** With weight decay in {0.3, 1.0}, 5/5 seeds grok at noise ≤ 0.10 and 0/5 seeds grok at noise ≥ 0.15. The transition occupies a single 5-percentage-point band.
2. **Weight decay is a second-axis cliff.** wd=3.0 prevents grokking entirely (the model cannot fit the clean training set); wd=0.3 and wd=1.0 grok up to a noise-cliff that *shifts slightly with wd*.
3. **Noise ≡ "model-collapse" contamination at matched rate.** At noise=0.15, random-label noise and temperature-warped collapse contamination produce statistically indistinguishable test-acc and Fourier-concentration distributions (n=5 each). The original "collapse is a distinct phenomenon" framing is **refuted by our own baseline**.
4. **Scarcity dissociation.** At 50% less training data, the model still groks and Fourier concentration is *higher* than at full data — while at 15% corrupted data it does not grok. This rules out "contamination = effective sample-size shrinkage" as the mechanism.

## Status (2026-05-10)

Toy phase: complete. Real-LM phase: in-flight (data-prep job died at SLURM time limit; resubmission queued). Mechanistic causal analysis: observational only — surgical-transplant rescue (Experiment A) is the next milestone. Threshold theory (Experiment C) has the empirical wd × noise grid; closed-form derivation pending.

See `AUDIT_CLAUDE.md` for the most recent independent on-disk audit and `NEXT_STAGE.md` for the week-by-week plan.

## Quick start

```bash
# Install (uv + venv, NOT conda — we are on Rivanna)
uv venv .venv && source .venv/bin/activate
uv pip install torch numpy matplotlib scipy

# Single training run
python src/train.py --condition pure --max-steps 50000

# Reproduce the wd × noise grid (SLURM array, 90 jobs)
sbatch slurm/exp_c_grid.sbatch

# Reproduce the noise / scarcity / multi-seed baselines
sbatch slurm/baselines.sh

# Surgical transplant rescue (after Experiment A is run)
python src/transplant_rescue.py \
    --pure-run results/exp_c_grid/wd1/noise0/seed_42 \
    --contam-run results/exp_c_grid/wd1/noise0.15/seed_42 \
    --output-dir analysis/transplant
```

## Architecture

- 1-layer Transformer encoder, d_model=128, 4 heads, d_ff=512 (~214K params).
- Token + positional embeddings, GELU FFN, mean-pool over the two input positions, p-way output head.
- Default optimizer: AdamW, lr=1e-3, wd=1.0, batch=512, 50000 steps.
- Default task: `(a+b) mod 59` with 30% train fraction.

## Repository layout

```
src/
  train.py                 # core training loop
  data.py                  # dataset generation + collapse + label noise
  model.py                 # ModularArithmeticTransformer (214K params)
  run_exp_c_grid.py        # wd × noise grid driver
  run_grid.py              # collapse-level × severity grid
  run_noise_baseline.py    # uniform-label-noise baseline
  run_scarcity_baseline.py # train-fraction baseline
  run_multi_seed.py        # 5-seed × 5-condition repeat
  run_prime_sweep.py       # second-prime brittleness check (NEW)
  transplant_rescue.py     # surgical-circuit transplant (Exp A, NEW)
  threshold_theory.py      # closed-form η*(λ, p, d) + empirical fit (Exp C, NEW)
  causal_circuit_rescue.py # observational per-matrix rank trajectory
  progress_measures.py     # Chan-style progress measures, leading-indicator variant
  analysis.py              # per-condition plot/table generation
  contamination/           # toy contamination experiment
  contamination_real/      # GPT-2 medium + LoRA on contaminated OWT (Exp B)

slurm/                     # one .sbatch per experiment block
analysis/                  # generated tables, plots, markdown summaries
results/                   # results.json + checkpoint_*.pt per run
```

## Analysis Pipeline

The comprehensive analysis pipeline has been extended to study the interplay between model collapse and grokking:

1. **Results Aggregation & Report Generation**: Summarize JSON logs into pandas DataFrames and generate publication-ready plots (Accuracy vs Steps, Fourier Concentration) using `src/analysis/results_aggregator.py`.
2. **Weight Trajectory Visualization**: Visualize the overall reduction in weight norms during training, annotating exact grokking transition points (`src/analysis/weight_trajectory.py`).
3. **Phase Transition Detection**: Automatically identify precise grokking boundaries (when test accuracy crosses and remains above a threshold) and detect phase shifts in Fourier concentration from uniform to concentrated states (`src/analysis/phase_detector.py`).
4. **Attention Pattern Evolution**: Extract attention weights directly from saved model checkpoints to study how attention allocation (e.g., pos 0 vs pos 1) changes before, during, and after grokking, comparing patterns across varying levels of model collapse (`src/analysis/attention_analysis.py`).

### Visualization Guide

To reproduce the analysis plots from trained models, use the provided tools:

```bash
# 1. Aggregate results and plot accuracy/fourier trajectories
python -c "from src.analysis.results_aggregator import plot_aggregated_results, generate_markdown_report; from pathlib import Path; plot_aggregated_results(Path('results'), Path('analysis/output')); generate_markdown_report(Path('results'), Path('analysis/output/report.md'))"

# 2. Plot weight norms per layer across conditions
python -c "from src.analysis.weight_trajectory import plot_weight_trajectories; from pathlib import Path; plot_weight_trajectories(Path('results'), Path('analysis/output/weight_trajectories.png'))"

# 3. Visualize attention evolution for the pure condition
python -c "from src.analysis.attention_analysis import plot_attention_evolution; from pathlib import Path; plot_attention_evolution(Path('results/pure'), Path('analysis/output/attention'))"

# 4. Compare attention patterns across all conditions at a specific step
python -c "from src.analysis.attention_analysis import compare_collapse_attention; from pathlib import Path; compare_collapse_attention(Path('results'), Path('analysis/output/attention_comparison.png'), step=50000)"
```

## References

- Power et al. (2022), *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*.
- Liu et al. (2022), *Omnigrok: Grokking Beyond Algorithmic Data*.
- Nanda et al. (2023), *Progress Measures for Grokking via Mechanistic Interpretability*.
- Chan et al. (2023), *Causal Scrubbing*.
- Shumailov et al. (2024), *The Curse of Recursion*.
- Dohmatob et al. (2024), *A Tale of Tails: Model Collapse as a Change in Scaling Laws*.
- Frei et al. (2022), *Benign Overfitting Without Linearity*.
