# NEXT_STAGE.md — Plan from 2026-05-10 audit forward

This plan is the operational follow-up to `AUDIT_CLAUDE.md` and
`RESEARCH_ROADMAP.md`. It is laid out week-by-week with concrete file edits
and SLURM submissions, **kill criteria** that would cause us to drop a
direction, and a target venue + drop-dead date.

Today: **2026-05-10**. Target submission: **NeurIPS 2026 MechInterp Workshop
(mid-July 2026)** as the safe landing for Experiments A + C; ICLR 2027
(mid-Sep 2026 deadline) as the full-A+B+C target if Experiment B partially
lands. Do **NOT** attempt NeurIPS 2026 main track (deadline 22 May, twelve
days from today, with three open-ended experiments unfinished).

---

## Week 1 (2026-05-10 → 2026-05-17) — Kill or commit

Goal: get Experiment A's first signal and Experiment C's empirical fit on
disk; reframe public-facing artifacts. After this week we either have
preliminary evidence the rescue works or we know it doesn't and pivot.

### Files / commands

- [x] `README.md` — drop "collapse vs noise" framing, anchor on "label-noise
  cliff + scarcity dissociation". Done.
- [x] `CLAUDE.md` — update project instructions to reflect the audit. Done.
- [x] `src/transplant/transplant_rescue.py` — real surgical patch + retrain + random
  control, paired-seed. Done.
- [x] `src/threshold_theory.py` — explicit derivation of η*(λ, p, d) and
  empirical fit on the existing `results/exp_c_grid/`. Done.
- [x] `src/run_prime_sweep.py` + `slurm/prime_sweep.sbatch` — second-prime
  brittleness sweep. Done.
- [x] `src/leading_indicator_test.py` — pre-registered AUC test for the two
  leading-indicator candidates (early Fourier slope, early log-‖W‖ slope).
  Done.
- [x] `slurm/real_generate_v2.sbatch` — resumable per-(ratio, seed) array
  replacing the cancelled v1 monolith. Done.
- [ ] **Submit:** `sbatch slurm/threshold_theory.sbatch` (CPU, 30 min).
- [ ] **Submit:** `sbatch slurm/transplant.sbatch` (GPU array 0–9, ~4 h each).
- [ ] **Submit:** `sbatch slurm/prime_sweep.sbatch` (GPU array 0–89%30, ~1.5 h
  each, ≈4.5 wall-clock h with 30 concurrent).
- [ ] **Submit:** `sbatch slurm/real_generate_v2.sbatch` (GPU array 0–17%4,
  ~6–8 h each, total wall-clock ~3 days at concurrency 4).

### Kill criteria for Week 1

- **Drop Experiment A** if all `transplant_<C>` zero-shot variants stay below
  `baseline_contam + 0.05` AND all `transplant_<C>+rt` variants stay below
  the rescue threshold (defined as `baseline_pure - 0.10`). Interpretation:
  no single-component patch shifts the model meaningfully — the failure is
  not localized.
- **Reframe Experiment C** if the bootstrap CI on the empirical exponent b
  contains 0 (i.e. η* doesn't depend on λ at all in our range). Then the
  theory section becomes "Why doesn't the cliff move with λ?" — interesting
  but a different paper.
- **Drop the prime sweep** if p=97 fails to grok at *all* noise levels in our
  range (would suggest p=59 is the only easy prime in our resolution).

### Definition of "we're on track"

By 2026-05-17 EoD:
- transplant: ≥1 component shows ≥0.20 absolute test_acc gain over
  `baseline_contam` in *zero-shot* mode for the wd1/n0.15 paired seeds.
- threshold theory: empirical b is finite and the 95% CI lies in [0.5, 1.5]
  (i.e. consistent with linear-in-λ to factor of 2).
- prime sweep: at p=97 the cliff exists between two noise levels (don't have
  to be the same as p=59 — just *exists*).
- real-LM data-gen: at least ratio_15/seed_0 has been generated.

---

## Week 2 (2026-05-17 → 2026-05-24) — Tighten and replicate

Goal: replicate Week 1 signals at scale; nail down the theory; push real-LM
training while compute is queued.

### Tasks

- **Replicate transplant rescue** at all 5 seeds × {n=0.15, n=0.20} (already
  in the array launched Week 1). Aggregate into `analysis/transplant/aggregate.md`
  with per-component mean and 95% CI rescue rate.
  - New file: `src/aggregate_transplant.py`. Reads all
    `analysis/transplant/wd1_n*_s*/rescue_results.json` and emits a single
    summary plot + table.
- **Refine threshold theory** if the Week-1 fit is inconclusive: launch a
  finer-grained noise sweep `noise ∈ {0.06, 0.08, 0.10, 0.12, 0.14, 0.16}` at
  wd ∈ {0.3, 1.0} × 5 seeds = 60 runs.
  - New SLURM: `slurm/exp_c_grid_fine.sbatch`. Reuse `run_exp_c_grid.py`.
- **Train real-LM models** on the ratios that are ready:
  - `slurm/real_train.sbatch` already exists; resubmit with whatever ratios
    have produced datasets so far. Target: 3 seeds at each of {0%, 15%, 30%}
    by EoW.
- **Implement Fourier-basis projection patch** as Experiment A v2: instead of
  swapping whole matrices, project the contaminated MLP-out onto the
  pure-identified Fourier subspace and patch only that projection. New file:
  `src/transplant_basis_projection.py`. Cleaner mechanistic claim if it works.

### Kill criteria for Week 2

- **Drop the basis-projection variant** if matrix-level transplant already
  hits ≥80% of pure's test_acc with one component swapped — basis projection
  is redundant in that case.
- **Pause real-LM training** if the first finished real-LM training run shows
  no perplexity difference between ratio_0 and ratio_30 (within seed
  variance). The toy signal hasn't transferred; spend time on A+C instead.

---

## Week 3 (2026-05-24 → 2026-05-31) — Write up Section 3, 4, 5

Goal: write the paper sections corresponding to the empirical results we have
in hand. Hold §6 (real-LM) for Week 4.

### Tasks

- **Draft `paper/section_3_cliff.tex`** — the noise/collapse/scarcity 2-panel
  figure (already in `analysis/comprehensive_overview.png`), the
  noise-collapse equivalence statistical test, the scarcity dissociation.
- **Draft `paper/section_4_rescue.tex`** — the transplant table from Week 1–2
  (`analysis/transplant/aggregate.md`), specificity controls, basis-projection
  variant if it landed.
- **Draft `paper/section_5_theory.tex`** — the derivation in
  `src/threshold_theory.py` lifted into LaTeX, the empirical fit, the
  prime-sweep cross-validation as a free robustness check.
- **All-experiments rerun for the appendix**: run the same wd × noise grid
  with `train_fraction ∈ {0.2, 0.3, 0.4}` and `d_model ∈ {64, 128, 256}` to
  show robustness. ~150 runs, ~5 GPU-days at concurrency 30.

### Kill criteria for Week 3

- **Compress to a workshop paper now** if any of:
  - Transplant rescue at full 5-seed CI is < pure - 0.20 across all components
    (rescue exists but is weak).
  - Theory exponent CI does not include 1 *and* the data shows no obvious
    refined-model that fits.
  - Real-LM ratio_30 shows no detectable signal at 3 seeds.

---

## Week 4 (2026-05-31 → 2026-06-07) — Real-LM signal or honest null

Goal: decide whether Experiment B is in the paper or in an appendix.

### Tasks

- **Run mechanistic metrics on whatever real-LM training has finished.**
  Already plumbed in `src/contamination_real/mechanistic_metrics.py`.
- **Apply leading-indicator AUC test from `src/leading_indicator_test.py` to
  real-LM training trajectories** (port the same window/slope logic; reuse
  whichever metrics from `mechanistic_metrics.py` look stable across seeds).
- **External validation pass:** apply the geometry signal to the released
  Phi-3, Mistral 7B, Pythia 7B checkpoints (CPU/single-GPU, ~10 GPU-h total).
  - New file: `src/contamination_real/probe_released_models.py`. Stub now.

### Kill criteria for Week 4

- If geometry doesn't lead perplexity by ≥500 steps on any metric in any seed,
  Experiment B is reported as a *negative result* in the paper appendix
  with the honest framing "geometry does not appear to lead perplexity at
  GPT-2-medium scale on contaminated OWT."
- If the released-model probe gives an inconsistent ranking (Phi-3 not
  flagged, Pythia flagged), drop the external-validation paragraph.

---

## Beyond Week 4 — submission

**Workshop submission target: 2026-07-15 (NeurIPS Workshop deadline,
approximate).**

| Week | Deliverable |
|---|---|
| 5 (06-07 → 06-14) | First full draft, internal pass |
| 6 (06-14 → 06-21) | Plot polish, appendix completion, code repo cleanup |
| 7 (06-21 → 06-28) | Co-author / advisor review |
| 8 (06-28 → 07-05) | Revisions, second pass |
| 9 (07-05 → 07-12) | Final formatting |
| 10 (07-12 → 07-15) | **Submit.** |

If Experiment B lands convincingly by 06-14, revise target to **ICLR 2027**
(deadline 2026-09-15), with workshop submission as a parallel scoping
exercise. Otherwise, submit workshop as-planned and continue B for ICLR.

---

## Top-level kill criteria (project-level)

If by **2026-05-31** all of the following are true, the project converts to a
workshop-only "negative result + dataset release" paper, no main-track attempt:

1. No single-component transplant rescues to within 0.10 of pure's test_acc
   even with retrain (Experiment A is observational, not causal).
2. The bootstrap CI on the threshold-theory exponent b is too wide to support
   *any* power-law claim ([0.0, 2.0] or wider).
3. Real-LM contamination signal is undetectable at GPT-2-medium scale across
   3 seeds at ratios {0%, 30%}.

This is a stop-loss, not a default outcome.

---

## Compute budget

Cumulative through 2026-07-15 (NeurIPS workshop deadline):

| Item | GPU-h |
|---|---|
| transplant_v1 (10 jobs × 4h) | 40 |
| prime_sweep (90 jobs × 1.5h) | 135 |
| theory analysis (CPU) | 0 |
| real-LM data-gen v2 (18 jobs × 6h) | 108 |
| real-LM training (15 runs × 6h) | 90 |
| robustness appendix (150 runs × 0.5h) | 75 |
| **Total estimated** | **≈450 GPU-h** |

This fits within `zhangmlgroup` allocation given ~6 weeks of submissions.

---

## What's *not* on this plan

- **No new toy framing.** No more "collapse-vs-noise". Forward, we describe
  the corruption channel as "label-noise rate η" with a footnote that the
  temperature-warped collapse intervention reduces to it.
- **No new prime-fraction-d_model 3D sweep at full seed count.** Robustness
  appendix uses one-axis-at-a-time.
- **No Fourier-circuit causal-scrubbing reimplementation** unless transplant
  rescue lands and we need stronger evidence.
