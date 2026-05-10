# Independent Audit — Grokking Under Distributional Collapse

**Auditor:** Claude (read-only review)
**Date:** 2026-05-10
**Scope:** Source code, results JSON, analysis reports, training logs, git history, and Slurm job state on disk. No assumptions taken from README/CLAUDE.md beyond verifying they match the data.

---

## 1. What is the actual result?

The project trains a 1-layer transformer (213,947 params; d_model=128, 4 heads, d_ff=512) on `(a + b) mod 59` with 30% train fraction, AdamW, batch 512, 50,000 steps. The "collapse" intervention replaces a fraction of training targets with samples from a temperature-warped frequency distribution; a separate "noise" intervention replaces them with uniform random wrong labels. The model is held constant across all conditions; only the data corruption and `weight_decay` are varied.

### What has actually been run (verified on disk)

| Sweep | Conditions | Seeds | Total runs | Status |
|---|---|---|---|---|
| `seed_sweep/` (initial 5-condition) | pure, low_collapse, medium, high, severe | 1 (seed 42) | 5 | done |
| `multi_seed/` (5-condition × 5 seeds) | same 5 conditions | 5 (42–46) | 25 | done |
| `grid/` (level × severity) | 4 levels × 3 severities | 5 | 60 | done |
| `noise_baseline/` | noise ∈ {0, 0.05, 0.15, 0.3, 0.5} | 5 | 25 | done |
| `scarcity_baseline/` | train_fraction ∈ {0.15, 0.21, 0.255, 0.285, 0.3} | 5 | 25 | done |
| `exp_c_grid/` (wd × noise) | wd ∈ {0.3, 1, 3} × noise ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3} | 5 | 90 | done |
| `contamination/` (GPT-2 medium + LoRA on OpenWebText) | ratio ∈ {0, 10} | 1–3 | **4** | partial (only ratio_0×3 seeds + ratio_10×1 seed); data-prep job for higher ratios got cancelled at SLURM time limit |

Total toy runs on disk: **230**. Real-LM runs on disk: **4**.

### Key numbers from `analysis/exp_c_grid_summary.md` (verified against the underlying CSV)

**Grok rate** (fraction of 5 seeds reaching test_acc ≥ 0.95):

| wd \ noise | 0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| 0.3 | 1.00 | 1.00 | 1.00 | **0.00** | 0.00 | 0.00 |
| 1.0 | 1.00 | 1.00 | 1.00 | **0.00** | 0.00 | 0.00 |
| 3.0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Mean final test accuracy:**

| wd \ noise | 0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|
| 0.3 | 1.000 | 0.979 | 0.961 | 0.804 | 0.698 | 0.291 |
| 1.0 | 0.984 | 0.977 | 0.914 | 0.842 | 0.758 | 0.236 |
| 3.0 | 0.147 | 0.121 | 0.078 | 0.104 | 0.077 | 0.053 |

**Noise–collapse equivalence (the load-bearing baseline result):**
- noise_baseline @ noise=0.15, 5 seeds: final test_acc = 0.827 ± 0.045; Fourier = 0.177 ± 0.003
- collapse grid @ level=0.15, severity=0.6, 5 seeds: final test_acc = 0.823 ± 0.048; Fourier = 0.184 ± 0.009
- Statistically indistinguishable at n=5 each.

**Severity sweep at fixed level=0 and level=0.05:**
- All 15 seeds grok (5×3 severities each at L=0 and L=0.05). Severity has no measurable effect.

**Scarcity sweep:** All five `frac` settings (0.15–0.30) still grok; Fourier concentration is *higher* (0.30–0.51 reported in `comprehensive_summary.md`) than the pure 0.3-fraction baseline.

**Real-LM (`results/contamination/`):**
- 0% contamination: perplexity 111.9 ± 0.69 (n=3 seeds), attn effective rank 55.15 ± 0.42
- 10% contamination: perplexity 113.49 (n=1), attn effective rank 57.05
- Difference is within seed variance. **No usable signal** at this n.

**Causal circuit (Experiment A) status:** the existing `src/causal_circuit_rescue.py` is **not the rescue experiment described in the roadmap.** It computes per-matrix effective-rank trajectories and identifies which matrix has the largest single-step jump per run. It does *not* perform any transplant or surgical patch. The output (`analysis/causal_circuit/causal_circuit_summary.md`) is an **observational** identification of which matrix changes most across the grokking transition — different per run (token_embed for wd=0.3 clean, attn in-proj for wd=0.3 noisy, FFN linear2 for wd=1 both clean and partial-grok).

---

## 2. Is the result positive, negative, or inconclusive?

**Mixed, leaning negative on the original framing; one genuinely interesting empirical thread.**

### Positive (robust)
- The grokking cliff between noise 0.10 and 0.15 is sharp, replicable across 5 seeds and 2 weight-decay values, and the numerical pattern matches across the orthogonal "noise" and "collapse" formulations.
- The weight-decay sweep adds a second sharp boundary: `wd=3.0` prevents grokking entirely (train acc 0.15–0.27, test acc near chance), which is independent of corruption — the model can't even fit the clean training set when decay is too strong.
- The scarcity dissociation is real and non-trivial: at 50% of training data, the model still groks and Fourier concentration is *higher* than at full data. This rules out "contamination ≡ effective dataset shrinkage" and is the most distinctive finding in the project.

### Negative / refuted by the project's own data
- **The original framing — "model-collapse contamination is a distinct phenomenon from random label noise" — is refuted by the noise_baseline.** At matched corruption rate, the two interventions produce statistically indistinguishable test-accuracy and Fourier-concentration distributions. This is acknowledged in `RESEARCH_ROADMAP.md` §2 ("disproved by our own noise baseline").
- **Severity is empirically irrelevant.** At fixed level, the temperature-warped corruption shape has no detectable effect on outcome. So the data-side contribution collapses to "label-noise rate matters" — a known result.
- **Fourier concentration is a lagging, not leading, indicator** on this task. The `seed_sweep_report.md` §C documents that at the grokking step, pure and low-collapse have nearly identical Fourier values (0.198 vs 0.202); the metric jumps *after* test accuracy does. This contradicts framing it as a "progress measure" in the predictive sense.

### Inconclusive
- The real-LM extension. Only 4 runs, only 2 ratios, only 1 seed at the contaminated ratio. The two ratios are 0% and 10% — both still below the 5–10% cliff seen on the toy. The data-prep Slurm job that would have generated ratios up to 50% was cancelled at time limit (`logs/real-gen-12708261.err`, last line: "CANCELLED ... DUE TO TIME LIMIT"); no follow-up training has produced new data.
- The causal-circuit claim. The current analysis is correlational ("which matrix changes most"). The actual transplant rescue described in the roadmap has not been run. So statements like "wd shifts the noise tolerance of the same circuit" are at best a hypothesis consistent with the observational data, not demonstrated.
- The theory. The roadmap proposes a closed-form prediction of η* in terms of (λ, p, d). Only the empirical wd-sweep half exists; no derivation is in the repo.

---

## 3. What is novel vs prior work?

The project itself enumerates the prior work in `RESEARCH_ROADMAP.md` §2 — the audit confirms each citation maps to a published claim that subsumes a portion of this work.

**Not novel:**
- Label noise prevents grokking (Power 2022, Liu 2022 *Omnigrok*, Anil 2022).
- Fourier-top-k concentration tracks generalization in modular arithmetic (Nanda 2023, Chan 2023).
- Effective rank / weight norm fall during the grokking cleanup phase (Liu 2022, Chan 2023).
- Synthetic-data contamination degrades models (Shumailov 2024).
- Phase transitions exist in synthetic-data ratio (Dohmatob 2024, Gerstgrasser 2024).

**Plausibly novel (but small):**
- The clean joint **scarcity-dissociation finding**: at 50% less data the model still groks and Fourier concentration is *higher*, while at 15% corrupted data it does not grok. Together this rules out the "contamination = less effective data" interpretation. This empirical dissociation is the most distinctive thing in the repo. It is also the kind of negative-result finding that is publishable as a section but probably not as a paper on its own.
- The **`wd × noise` 2D heatmap** showing that wd=3 fails *independent* of noise (the model can't memorize), wd=0.3/1.0 grok up to a cliff that shifts slightly with wd (cliff at noise=0.10 for wd=1, noise=0.05 for wd=0.3 by the Fourier criterion). The wd=3 catastrophic regime is a useful "too-much-regularization" boundary that complements the noise boundary — but again it's likely a corollary of generic AdamW behavior, not a new mechanistic phenomenon.

**Refuted by the project's own data:**
- The headline framing in the README ("collapse simulates model collapse and we expect it to differ from label noise"). The matched-rate comparison shows it does not.

---

## 4. Biggest open questions

1. **Does any of this hold at LM scale?** The real-LM track has 4 runs total. The single contamination data point is below the toy cliff. This is the question that determines whether the paper is "toy-only workshop" or "main-track."
2. **Does an actual surgical transplant rescue grokking?** The roadmap's Experiment A is unrun. Current "causal" analysis is observational. Without a real transplant, the mechanistic story stays correlational.
3. **Is the cliff position predicted by a closed-form?** The theory half of Experiment C is not in the repo. Without it, the empirical wd-sweep result is "the cliff shifts with wd" — descriptive, not predictive.
4. **Is `p=59` doing the heavy lifting?** No replication at a second prime. The standard grokking literature (Power, Nanda) found `p=113`/`p=97` corners are also robust, but the *exact* numerical cliff position may not be — the project should at minimum run one alternate prime to argue against brittleness.
5. **Are noise and collapse really equivalent at the *distribution* level, or only at the *outcome* level?** They are matched on final test acc and Fourier concentration, but no distribution-level diagnostic (e.g., loss landscape geometry, gradient noise statistics) has been computed to confirm they are the same intervention. The "they look the same" claim relies on a 4D summary collapse.
6. **What is the `wd=3` regime actually doing?** Train acc 0.15–0.27 with Fourier concentration 0.3–0.7 (highest in the grid!) suggests the model is doing *something* structured, just not fitting the data. Could be a useful negative control or a clue about what direction weight decay pushes representations toward.

---

## 5. What should happen next?

In priority order, based on what the data already shows and what's missing:

1. **Reframe the writeup, immediately.** The README and the early CLAUDE.md still pitch "collapse vs noise." The roadmap already concluded this framing is fatal. Drop it. The defensible framing is: *"Label-noise rate determines a sharp grokking cliff; weight decay shifts the cliff and beyond a threshold prevents grokking entirely; the scarcity baseline dissociates noise from sample-efficiency."*
2. **Restart the real-LM data pipeline with proper wall-clock allocation.** The current `real-gen-12708261` job died at time limit before producing ratios 15/30/50. Without those, the entire Experiment B story has 1 contaminated seed at the wrong ratio.
3. **Actually run the surgical-transplant version of Experiment A.** The infrastructure (`src/causal_circuit_rescue.py`) reads checkpoints and computes per-matrix rank changes — the missing piece is the transplant + retrain loop described in the roadmap. Without it, Section 4 of the planned paper is observational.
4. **Add a second prime replication.** A 30-run sweep at p=97 (5 seeds × {wd=1} × 6 noise levels) is ~1 GPU-hour and answers the brittleness question. Cheap; high-impact for credibility.
5. **Derive (or at least bound) η* analytically.** The empirical wd-sweep half of Experiment C is done. The theory is not. Even a non-rigorous order-of-magnitude argument that predicts cliff shift with `√λ` would let the paper claim "predicted vs. measured."
6. **Audit `compute_fourier_concentration`'s lagging behavior.** The metric is well-defined but lags the behavioral transition by ~100–500 steps in this setup. Either (a) find a leading variant (e.g., slope-during-memorization, which the seed-sweep report mentions but doesn't pre-register), or (b) drop the "progress measure" claim and present it as a post-grok mechanistic signature.

---

## 6. Bottom line

This is a **competent, thoroughly documented toy study** with internally honest analysis. The project has produced ~230 toy runs, 4 real-LM runs, and a complete analysis pipeline. The infrastructure is solid; the data are real and reproducible from the saved `results.json` files.

The **original novelty claim is refuted by the project's own baseline.** The roadmap acknowledges this and proposes three salvage paths (A: causal rescue, B: real-LM extension, C: theory). On disk, A is observational not causal, B is 1 contaminated seed at 10% with the data-prep job dead at time limit, and C is empirical-only with no theoretical derivation.

**What remains, as-is, is publishable at workshop scale**: the wd × noise grid with the noise-collapse equivalence and the scarcity dissociation, framed as "label-noise rate is the causal driver of grokking cleanup-phase failure, and we falsify two natural alternative hypotheses (synthetic-data-as-distinct-phenomenon, contamination-as-sample-efficiency)." Main-track requires *at least one* of A (real transplant), B (real-LM cliff demonstrated), or C (closed-form prediction) to land. None of those has landed yet.

The author's own assessment in `RESEARCH_ROADMAP.md` matches this audit: "Do not submit the current results to a main track without at least A and C." I concur.
