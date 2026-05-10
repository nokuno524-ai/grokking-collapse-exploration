# Seed Sweep Analysis — Grokking Under Distributional Collapse

**Sweep:** `results/seed_sweep/` (Slurm job 12689523)
**Wall time:** 12 min 12 s on a single GPU (5 conditions × 50 000 steps each)
**Seed:** 42 (single seed across all conditions)
**Model:** 1-layer transformer, d=128, 4 heads, d_ff=512, 213 947 parameters
**Task:** `(a + b) mod 59`, 30% train fraction, AdamW with `lr=1e-3`, `weight_decay=1.0`, batch size 512
**Eval cadence:** every 100 steps; checkpoints every 5 000 steps (10 per condition)

The "collapse" intervention replaces a fraction `collapse_level` of training targets with samples from a temperature-warped frequency distribution, where temperature is set by `collapse_severity` (higher severity ⇒ flatter, more incorrect distribution). See `src/data.py::apply_collapse`.

---

## A. Per-condition summary

| Condition | `collapse_level` | `collapse_severity` | Grokked? | Grokking step | Final train acc | Final test acc | Best test acc (step) | Final Fourier conc. | Final ‖W‖ | Final embedding rank |
|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|---:|
| **pure** | 0.00 | 0.5 | ✅ | **1 700** | 1.0000 | **1.0000** | 1.0000 (1 900) | **0.301** | 30.31 | **25.13** |
| **low_collapse** | 0.05 | 0.3 | ✅ | **2 800** | 1.0000 | 0.9848 | 0.9947 (13 000) | 0.205 | 36.32 | 32.34 |
| **medium_collapse** | 0.15 | 0.5 | ❌ | — | 1.0000 | 0.8252 | 0.8900 (33 900) | 0.185 | 43.82 | 37.78 |
| **high_collapse** | 0.30 | 0.7 | ❌ | — | 0.9875 | 0.2544 | 0.2995 (40 700) | 0.166 | 53.52 | 36.04 |
| **severe_collapse** | 0.50 | 0.9 | ❌ | — | 1.0000 | 0.0464 | 0.0739 (1 500) | 0.120 | 55.37 | 37.52 |

Grokking threshold: `test_acc ≥ 0.95` (first crossing).

**Important caveat: collapse_severity is not held constant across conditions.** The sweep varies `collapse_level` *and* `collapse_severity` simultaneously (0.3 / 0.5 / 0.7 / 0.9 for low/medium/high/severe), so the conditions confound *fraction of corrupted labels* with *how badly each label is corrupted*. The clean comparisons in this report should therefore be read as effects of the joint factor, not of `collapse_level` alone.

This run also differs from the result table in `CLAUDE.md` (which lists pure grokking at step 1 400, low at step 3 100, medium 83.9% / Fourier 0.170, etc.). Those numbers are from an earlier RTX 5080 run; the current sweep was produced after the seeding / DataLoader / circuit-onset fix in `450541d`. The qualitative finding ("collapse kills grokking above ~10% corruption") survives, but the exact numbers have shifted.

---

## B. Grokking dynamics

### Pure (grokked, step 1 700)
Textbook delayed-generalization signature.
- `train_acc ≥ 0.99` at step **900**, with `test_acc ≈ 0.28` (memorization complete).
- Slow climb — `test_acc` 0.28 → 0.85 between steps 900 and 1 500.
- Sharp transition: 0.85 → 0.93 → **0.997** across steps 1 500 → 1 600 → 1 700 (~12 percentage points per 100-step eval window).
- Memorization-to-grokking delay: **800 steps**.
- After grokking, `test_acc` stays ~1.0 for the remaining 48 300 steps with occasional brief dips (e.g. step 6 500: 0.964) that recover within one eval window.

### Low collapse, 5% (grokked, step 2 800)
- Memorized at step 1 100 (`test_acc ≈ 0.18`).
- Less sharp than pure: the climb from 0.80 → 0.95 takes ~600 steps (steps 2 200–2 800) vs. ~200 in pure.
- Memorization-to-grokking delay: **1 700 steps** — over 2× longer than pure.
- After "grokking," `test_acc` *oscillates* between 0.91 and 0.99 for tens of thousands of steps and lands at 0.985 — never as locked-in as pure. Best ever was 0.9947 at step 13 000.

### Medium collapse, 15% (no grok)
- Memorized at step 1 500 (`test_acc ≈ 0.50` — already partial generalization on the clean labels).
- Slow, monotonic creep: 0.50 → 0.66 (5k) → 0.71 (10k) → 0.79 (25k) → **peak 0.890 at step 33 900** → drifts down to 0.825 by step 50k.
- Behaves like *partial circuit formation that stalls.* The model is trying to grok and not failing catastrophically, but never breaks the 95% barrier in the budget.

### High collapse, 30% (no grok)
- Train accuracy itself struggles: only reaches 0.988 by the end (vs. 1.0 elsewhere) — the corrupted labels are inconsistent enough that fitting them perfectly is hard with `weight_decay=1.0`.
- Test accuracy is **stuck at chance + small offset**: 0.17 → 0.20 → 0.27 → 0.30 (peak at step 40 700) → 0.25.
- Pattern looks like noisy memorization without circuit formation. No transition.

### Severe collapse, 50% (no grok)
- Memorized noise at step 1 400 (training fits the corrupted labels), but `test_acc` is essentially 1/p ≈ 0.017 with brief excursions to 0.07 in the early steps.
- **Test accuracy is monotonically near-flat throughout — and slightly worse than its peak at step 1 500** (0.074), meaning later training mildly *hurts* generalization.
- This is the cleanest "no signal" condition and serves as a useful negative control.

---

## C. Mechanistic analysis

### Fourier concentration (top-5 frequencies, embedding spectrum)

Trajectory at memorization → at grokking → final:

| Condition | At memorization | At grokking | Final | Δ(final − mem) |
|---|---:|---:|---:|---:|
| pure | 0.159 | 0.198 | **0.301** | +0.142 |
| low_collapse | 0.169 | 0.202 | 0.205 | +0.036 |
| medium_collapse | 0.135 | — | 0.185 | +0.050 |
| high_collapse | 0.127 | — | 0.166 | +0.039 |
| severe_collapse | 0.105 | — | 0.120 | +0.015 |

The Fourier concentration ordering at the end mirrors the test-accuracy ordering exactly (Spearman ρ = 1.0 across the 5 conditions). It's a **lagging** rather than leading indicator in this setup: at the moment grokking is detected, pure is at 0.198 and low is at 0.202 — essentially the same value, so concentration crossing ~0.2 corresponds to the *behavioral* transition rather than preceding it. Concentration continues climbing for ~5 000 steps after grokking in pure (0.198 → 0.307 by step 10 000), suggesting cleanup phase happens largely *after* the test-accuracy jump.

### Weight norm

All conditions follow a "rise-then-fall" arc, but the depth and timing differ markedly:
- **pure:** ‖W‖ peaks at **55.7 (step 6 500)**, decays to ~30 by end. Big, late drop (cleanup).
- **low:** peaks at **55.7 (step 7 000)**, decays to 36.
- **medium:** peaks at **62.5 (step 4 800)**, decays to 44.
- **high:** peaks at **68.7 (step 7 800)**, decays only to 53.
- **severe:** peaks at **64.0 (step 8 100)**, decays to 55.

A clean monotonic trend: **conditions that grok end at lower weight norm.** Weight decay is doing its job in the clean cases — it shrinks the memorization-phase weights down to whatever the grokked solution actually needs. In corrupted cases the decay still pulls, but the loss landscape pushes back, leaving the model parked at a higher norm.

### Embedding rank (effective rank)

- pure ends at **rank 25.1** — close to 2k+1 = 11 expected for k=5 used Fourier frequencies, but elevated by orthogonal noise; still by far the lowest.
- low ends at 32.3.
- medium / high / severe all hover around 36–38.

Embedding rank looks like the cleanest single mechanistic signature in this sweep: it's a monotonic measure of how compressed the embedding's linear structure has become, and only pure and (to a lesser extent) low collapse exhibit substantial compression.

### Phase-transition signature?

Within the pure run, the order of milestones is:
1. Weight norm rises (steps 0–700).
2. Memorization completes (step 900).
3. Fourier concentration starts compounding (steps 900–1 500: 0.16 → 0.18, slow).
4. **Test accuracy jumps** (steps 1 500–1 700: 0.86 → 0.93 → 0.997).
5. Weight norm collapses, embedding rank drops, Fourier concentration grows further (steps 1 700–10 000).

Concretely, **none of the three mechanistic metrics show a sharp pre-grok inflection in this 100-step-resolution log**. They are smooth where test accuracy is sharp. The cleanest *predictive* signal in this sweep would be the *slope* of Fourier concentration during memorization (pure: ~0.005/100 steps over 900–1 500; low: ~0.003; medium: barely positive; severe: flat) — but that requires more than one seed to make load-bearing.

---

## D. Statistical robustness

This is **N=1 per condition.** Any conclusion about ordering or thresholds rests on a sample of one.

For a 5-seed (or 10-seed) version we'd want:
- **Seeds 42, 43, 44, 45, 46** (or any 5 disjoint values) for each of the 5 conditions → 25 runs.
- **Per-run cost on RTX 5080-class hardware:** ~2 min 30 s wall (12:12 / 5).
- **Sequential cost:** 25 × 2.5 min ≈ **63 min**.
- **Parallel via Slurm array job (`--array=0-24`)** on Rivanna's RTX 6000 / 4000 Ada partitions: bottleneck is queue wait, not compute — a single A100 or RTX 6000 can finish each in ≤5 min, so wall time is roughly *queue time + 5 min*.
- Memory: each run used ≤32 GB host + ≤2 GB GPU; trivially fits anywhere.
- **What to report per condition:** mean ± 95% CI on (final test acc, grokking step, final Fourier conc., final weight norm). For grokking step, report median + IQR — distribution is heavy-tailed in the failing conditions (often `None`).
- **What to test statistically:** (a) Mann-Whitney U on grokking step, pure vs. low; (b) two-sample t-test on final test accuracy between adjacent severity levels with Holm-Bonferroni correction across 4 comparisons.

To strengthen claims further:
- **Decouple the confound** between `collapse_level` and `collapse_severity`. Run a 2D grid: e.g. `level ∈ {0, 0.05, 0.15, 0.30}` × `severity ∈ {0.3, 0.6, 0.9}` × 5 seeds = 60 runs ≈ 2.5 hours sequential.
- **Sweep the prime** — the 0.59 prime / 30% train fraction / `wd=1.0` corner is known to grok robustly. Reproducing the trend at p=97 or p=113 would rule out brittleness to this exact toy.

---

## E. Publication readiness

**As-is: not publishable.** The qualitative finding is consistent with intuition (label noise breaks toy grokking) and the experiment is clean, but:

1. **Single seed.** No reviewer accepts a 5-point trend with no error bars in 2026.
2. **Confounded sweep.** Severity and level co-vary, so the dose-response curve is not interpretable as "X% corruption causes Y% accuracy drop."
3. **No baseline ablations.** Compared to *random label noise* with the same fraction? Compared to *fewer training examples* (data scarcity, not corruption)? These are needed to argue that "collapse" ≠ "noise" ≠ "less data."
4. **The mechanistic story is preliminary.** The metrics correlate with grokking but the run does not show the metrics *predict* grokking ahead of the behavioral transition, which is the load-bearing claim of the Chan et al. progress-measures literature this builds on.

**With a 5-seed × 4-level × 3-severity grid + a label-noise baseline + a data-scarcity baseline,** this becomes a credible workshop submission — likely scope: an ICML / NeurIPS workshop on data quality, model collapse, or mechanistic interpretability (e.g. ICML MechInterp workshop, NeurIPS workshop on Synthetic Data). With a 2-prime replication and a more careful story about the *mechanism* of collapse failure (e.g. spectral analysis of the corrupted target distribution, plus showing pure-data cleanup phase still happens but circuit formation is blocked), it could be a short paper.

**It is not a main-track ICML/NeurIPS contribution** because the underlying claim ("noisy labels hurt grokking") is incremental on Power et al. 2022 and Chan et al. 2023; what would make it main-track is a *mechanistic explanation of why* — e.g. showing that collapse breaks specific Fourier circuits in a predictable, severity-dependent way, ideally tied to a quantitative model of the failure.

---

## F. Key plots for a paper

1. **Fig. 1 — Headline: test accuracy vs. step, all conditions, 5 seeds.**
   - x: step (log scale, 100–50 000); y: test accuracy [0, 1].
   - 5 lines, mean ± 95% CI shading.
   - Annotate the grokking step with a vertical bar where applicable.

2. **Fig. 2 — Three-panel mechanistic trajectory (pure run, single seed for clarity).**
   - Panels (a) Fourier concentration, (b) weight norm, (c) embedding rank. Same x-axis (step, log scale). Vertical lines for memorization (`train_acc=0.99`) and grokking (`test_acc=0.95`).
   - Sub-figure (a) overlays test accuracy on a secondary y-axis to show the temporal ordering.

3. **Fig. 3 — Mechanistic-metric heatmaps across conditions.**
   - 3 heatmaps side-by-side: rows = condition (severity-ordered), columns = step (binned every 1 000 steps), color = metric value. One heatmap per metric (Fourier, ‖W‖, rank).
   - Reveals the "shoulder" where pure/low diverge from medium/high/severe.

4. **Fig. 4 — Final Fourier concentration vs. final test accuracy.**
   - x: final Fourier concentration; y: final test accuracy. One point per (condition, seed), color-coded by condition.
   - Shows the (probably linear) coupling between mechanistic and behavioral measures.

5. **Fig. 5 — Generalization gap evolution.**
   - x: step (log); y: `test_loss − train_loss` (log).
   - Pure should show the canonical "huge gap, then collapse"; corrupted conditions should show "huge gap, never collapses."

6. **Fig. 6 — Dose-response curve (requires the 2D grid run, not yet collected).**
   - x: `collapse_level` (0–0.5); y: P(grok within 50 000 steps), with CIs from seed variability. One line per `collapse_severity`. Identifies the threshold.

7. **Fig. 7 — Embedding-spectrum visualization at three checkpoints (memorization / grokking / final), pure vs. medium.**
   - Bar plot of average Fourier amplitude per frequency. Shows the top-5-frequency concentration emerging in pure but not in medium. This is the "money figure" for the mechanistic claim.

---

## Reproducibility pointers

- Source: `src/{train.py, model.py, data.py, progress_measures.py}`
- Per-step history is logged at 100-step resolution → `results.json` for each condition (5 029 lines, ~1.4 MB each).
- 10 model checkpoints per condition under `results/seed_sweep/<condition>/checkpoint_*.pt` enable re-running the mechanistic analysis (or running ones not collected here, e.g. logit-lens, attention-head ablations, full Chan-style "excluded loss") without retraining.
- Note: `progress_measures.py` currently computes the *generalization gap* and not Chan et al.'s true *excluded loss* (which would need the model and dataset, not just history). The header docstring acknowledges this.
- Pinned commit: `450541d` (CUDA seed, DataLoader seeding, severity ordering, circuit-onset detection fix).
