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

## References

- Power et al. (2022), *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*.
- Liu et al. (2022), *Omnigrok: Grokking Beyond Algorithmic Data*.
- Nanda et al. (2023), *Progress Measures for Grokking via Mechanistic Interpretability*.
- Chan et al. (2023), *Causal Scrubbing*.
- Shumailov et al. (2024), *The Curse of Recursion*.
- Dohmatob et al. (2024), *A Tale of Tails: Model Collapse as a Change in Scaling Laws*.
- Frei et al. (2022), *Benign Overfitting Without Linearity*.

## Early warning signals

Can we predict whether a model will grok long before the grokking cliff in test accuracy occurs?
The early warning signals module (`src/analysis/early_warning`) investigates this by extracting precursor signals from training logs up to a specific step.

### Log Format Expected
The module ingests training logs structured as lists of dictionaries (e.g. the `history` array in `results.json`). Each entry must contain at least:
- `step`: The training step (integer).
- `train_loss`: The loss on the training set (float).
- `test_acc`: The accuracy on the validation/test set (float).
- `weight_norm`: The norm of the model weights (float).

Optionally, logs can include `grad_norm` for gradient statistics.

### Precursor Signals
The following signals are computed over a rolling window leading up to the evaluation step:
1. **Train loss plateau slope**: The slope of the linear fit to the recent training loss curve.
2. **Weight norm derivative**: The slope of the linear fit to the recent weight norm curve.
3. **Gradient norm statistics**: Mean and variance of the gradient norm (if available).
4. **Test accuracy variance & autocorrelation**: Variance and lag-1 autocorrelation of the test accuracy curve.
5. **Delayed generalization score**: A composite score inspired by critical transitions literature (`variance * max(0, autocorrelation)`), which identifies rising variance and rising autocorrelation before a bifurcation.

### Running the Predictor
To run the prediction experiments across all multi-seed runs, executing LOOCV logistic regression for will-it-grok and linear regression for when-it-groks:
```bash
python -m src.analysis.early_warning.predictor
```
This will generate `grokking_early_warning.md` and boxplots for each fraction in `analysis/early_warning/`.

### Caveats
- These early warning signals are strictly correlational. They identify *markers* that precede grokking under these specific training setups (e.g. modular arithmetic, fixed learning rate), but do not imply a causal mechanism.
- The sample size of runs (n=25 across the multi-seed directory) is extremely small. The 100% predictive accuracies achieved at early fractions should be interpreted cautiously and may not generalize broadly outside of this narrow experimental setup.
