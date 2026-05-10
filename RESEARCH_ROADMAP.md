# Research Roadmap — Grokking Under Distributional Collapse

**Author:** research-advisor review · **Date:** 2026-05-10
**Status of project:** Toy modular-arithmetic phase fully complete (5-cond × 5-seed + 4×3×5 grid + noise & scarcity baselines, 50/50). Real-LM pipeline (GPT-2 medium + LoRA on contaminated OpenWebText) plumbed and partially running (`real-gen` job 12708261 still alive at 10h, 1× ratio_pct=10 contamination training partially logged).

---

## 1. Current results — what we actually have (one paragraph)

On a 1-layer transformer (214K params) trained to grok `(a+b) mod 59`, label-replacing **contamination produces a sharp grokking cliff between 5% and 15% corruption that is robust across 5 seeds and three severity settings**. At 0%/5% contamination, 5/5 seeds grok with final test accuracy 0.98 and final Fourier-top-5 concentration ≈ 0.31/0.21; at 15%/30%/50% contamination, 0/5 seeds grok and Fourier concentration plateaus at ≈ 0.18/0.16/0.12. **Severity (how flat the wrong-label distribution is) has no measurable effect at fixed level**: L=0.0 with sev∈{0.3,0.6,0.9} gives 5/5 grok in all cells; L=0.05 with the same severities gives 5/5; L=0.30 gives 0/5. **Random label-noise baselines reproduce the collapse curve numerically to within seed variance** (noise=0.15 → test 0.827±0.045, Fourier 0.177±0.003 vs collapse L=0.15,S=0.6 → 0.823±0.048, 0.184±0.009). **Data-scarcity baselines do not** — even at 50% less data, 5/5 seeds grok, with Fourier concentration *higher* (0.30) than pure (0.31) and rising to 0.51 at intermediate fractions. The full-realism (GPT-2 medium + LoRA) pipeline is built; only one contamination ratio has produced a partial log.

---

## 2. The novel contribution — brutally honest

### What is **not** novel

| Claim | Already shown by |
|---|---|
| "Label noise breaks grokking on toy tasks" | Power et al. 2022 (data-fraction sweep); Liu et al. 2022 *Omnigrok*; Anil et al. 2022 |
| "Fourier-top-k concentration tracks generalization in modular arithmetic" | Nanda et al. 2023; Chan et al. 2023 (progress measures) |
| "Effective rank / weight norm decrease during the grokking cleanup phase" | Liu et al. 2022; Chan et al. 2023 |
| "Training on AI-generated text degrades models" | Shumailov et al. 2024 *The Curse of Recursion* |
| "Phase transition exists in synthetic-data ratio for LMs" | Dohmatob et al. 2024; Gerstgrasser et al. 2024 |

### What our data **claims** to add

1. A clean dose-response cliff between 5% and 15% contamination on a controlled toy.
2. A monotonic Fourier-concentration signal across the cliff (Spearman ρ=1.0, n=5).
3. A real-LM pipeline that *will* show whether the toy signal scales.

### Why those three claims are not sufficient for a top-venue paper, on their own

- **Our own noise baseline disproves the framing.** Random-label noise at 15% and "model-collapse" contamination at 15% have *statistically indistinguishable* test accuracy (0.827 vs 0.823) and Fourier concentration (0.177 vs 0.184) at 5 seeds. The "collapse" intervention is empirically equivalent to label noise rate on this task. **If we cannot tell collapse from label noise, the contribution reduces to "label noise breaks grokking," which is not novel.**
- **Severity has no effect.** Our data shows the *fraction* of corrupted labels matters, not the *quality* of corruption. This further collapses the story to "label noise rate."
- **Fourier concentration is a *lagging* indicator at our resolution.** The seed-sweep report (analysis/seed_sweep_report.md §C) is explicit: at the moment of grokking in pure (step 1700), Fourier=0.198; at the moment of grokking in low_collapse (step 2800), Fourier=0.202 — essentially equal, and *both* are below the post-grok cleanup level (~0.30). The metric crosses a threshold *concurrent with* the behavioral transition, not before it. So the headline "Fourier concentration tracks contamination" is a **post-hoc, not predictive,** result.
- **Toy-only.** 214K parameters, 1 layer, p=59. Reviewers ask: does this hold at scale, on real text, on real architectures?
- **No mechanistic claim that is causal.** Correlations between contamination, generalization, and Fourier concentration do not establish that contamination *kills grokking by suppressing Fourier circuit formation*. They are equally consistent with "contamination prevents the model from finding any structured solution, of which Fourier is one."

### The one genuinely interesting empirical finding in the data

**Scarcity ≠ contamination.** With *half* the training data, the model groks fine and develops *more* concentrated Fourier features (0.51 vs 0.31). With *15%* of the training data corrupted, the model fails to grok and concentration plateaus at 0.18. **This dissociation is real and not derivable from any prior result we know of.** It rules out the most natural alternative explanation ("contamination = effective dataset size shrinks") and forces a *qualitative* difference between "less data" and "wrong data." The mechanism isn't sample efficiency — it's that corrupted gradients actively block the cleanup phase.

This dissociation, *if extended to real LMs*, is the load-bearing novel claim available from the current setup.

---

## 3. Three experiments that could make this a top-venue paper

The project has three exits, in increasing order of risk and reward.

### Experiment A — Causal circuit rescue (highest leverage, medium risk)

**Question:** If we surgically reinsert the Fourier-circuit components from a *pure-trained* checkpoint into a *medium-collapse* model, does grokking get rescued?

**Why this matters:** It would be the first true *causal* mechanistic claim about grokking failure under contamination. "Contamination correlates with no grokking" → "Contamination *prevents* grokking via suppression of a specific circuit, and we can prove it by transplant."

**Design:**
- Identify the Fourier-circuit basis on a pure-grokked checkpoint following Nanda et al. (key/query subspaces aligned with cos/sin frequencies).
- Project a contaminated model's MLP output onto the union (pure ⊕ contam) basis and re-train only the contaminated components for 2k steps with the pure components frozen.
- Compare against (i) re-training all weights for 2k steps (full-budget control) and (ii) random-direction patching (specificity control).

**Cost on Rivanna:** ~6 hours wall on a single A100/RTX 6000. Each rescue run = 2k steps of retraining, ≈30 s. 5 seeds × 4 contamination levels × 3 patch conditions = 60 runs × 30 s = 30 min compute. The cost is in setup, not training.

**Risk assessment:**
- *Medium.* The rescue could fail because contaminated and pure models converge to *structurally different* optima (different frequency bases, different head specializations), not "broken vs. intact" versions of the same circuit.
- Mitigation: run on the *same* seed (so initialization is identical) — this isolates the contamination effect from the seed-driven choice of which Fourier basis the model selects.
- Reviewer-killer: if the rescue works only when transplanting the *whole* embedding+MLP, the mechanistic claim is weakened to "contamination breaks the model in many places at once," which is not interesting.

**Outcome envelope:**
- *Best case:* Surgical patch of MLP-out projection onto Fourier basis rescues 80%+ of pure test accuracy. → Mechanistic causal claim secured. Strong figure.
- *Likely case:* Partial rescue (50–70%); rescue degrades smoothly with contamination level. → Useful, fits a section.
- *Worst case:* No rescue, or rescue requires patching everything. → Negative result, undermines the mechanistic story.

### Experiment B — Real-LM extension and detection (highest novelty, high risk)

**Question:** Does a representation-geometry signal (effective rank / attention entropy / token-embedding spectral concentration) detect contamination in language-model fine-tuning, and does it inflect *before* perplexity does?

**Why this matters:** This converts the toy result into a *deployable diagnostic*. Existing contamination/MIA methods (Min-K%, DetectGPT, neighborhood smoothness, perplexity ratios) require *test-set probes* or *known clean reference models*. A geometry-based signal computed from the *training trajectory itself* would be a new class of diagnostic — and the field needs one (Shumailov 2024, Dohmatob 2024).

**Design — already in `src/contamination_real`:**
1. Finish the in-flight real-gen pipeline (job 12708261).
2. Train GPT-2 medium with LoRA on OpenWebText with contamination ratios ∈ {0%, 5%, 15%, 30%, 50%}, 3 seeds each = 15 runs (≈4–6 GPU-hours each on A100 = 60–90 GPU-hr total).
3. Log the existing `mechanistic_metrics.py` battery (rank, attention entropy, feature density, gradient cosine, n-gram diversity, LoRA norm drift) every 500 steps + perplexity on a held-out OpenWebText slice.
4. **The pre-registered claim to test:** the *slope* of effective-rank decline crosses a critical-fraction threshold *N steps before* held-out perplexity does. Fit a change-point detector to each metric independently and compare change-point steps; report whether geometry leads perplexity by a statistically significant margin (paired Wilcoxon across seeds).
5. **External validation:** apply the geometry signal to *publicly released* models with documented contamination histories — e.g., Phi-3 (synthetic-heavy), Mistral 7B (curated), Pythia (clean). Compute the signal at the released checkpoints and predict a contamination ranking; cross-check against `Min-K% Prob` (Shi 2023), MIA AUC, and the "Pile-test memorization" benchmarks.

**Cost on Rivanna:** ~80 GPU-hr for 15 training runs (already plumbed). The publicly-released-model evaluation is CPU/single-GPU inference, ~10 GPU-hr.

**Risk assessment:**
- *High.* Three failure modes:
  1. The signal does not appear at GPT-2-medium scale (too few parameters relative to OpenWebText complexity).
  2. The signal appears, but *concurrent with* perplexity rather than leading it. (This is in fact what we already saw on the toy — Fourier was lagging.)
  3. The signal generalizes to fine-tune contamination but not to *pretraining-data* contamination, in which case the practical claim about published models fails.
- Mitigation: ablate model scale (GPT-2 small / medium / large) on the same data so the scale dependence is visible.

**Outcome envelope:**
- *Best case:* Geometry leads perplexity by 1k–10k steps; ranks Phi-3/Mistral/Pythia in the expected order on contamination. → NeurIPS-class result. The paper sells itself.
- *Likely case:* Geometry tracks perplexity tightly but doesn't lead it; ranks third-party models inconsistently. → Workshop paper + negative-result section.
- *Worst case:* No coherent signal at scale. → Toy-only paper (workshop).

### Experiment C — Theory: a noise-rate model of the cliff (highest rigour, lowest risk)

**Question:** Why is the cliff at 5%–15% specifically, and why doesn't severity matter?

**Conjecture:** With AdamW + weight decay = 1.0, the grokking cleanup phase is a balance between (a) the loss-driven gradient toward the clean memorization solution and (b) the weight-decay pull toward zero. Random label noise at rate η inflates the loss-gradient noise floor by O(η · ‖logit-grad‖). Above a critical η*, the noise floor is comparable to the cleanup gradient and the model gets parked at the memorization minimum. *Severity doesn't matter because the cleanup gradient is sensitive to any non-clean target, regardless of how non-clean.*

**Why this matters:** A theoretical prediction of η* in terms of (weight decay, learning rate, model dimension, prime size) would (i) explain why severity is irrelevant, (ii) explain the universality across noise-vs-collapse, and (iii) make a quantitative prediction we can test by sweeping weight decay or model size.

**Design:**
- Adapt the Frei et al. 2022 / Bai et al. 2021 framework for label-noise + weight-decay convergence to the explicit modular-arithmetic Fourier-feature model used by Nanda et al. 2023.
- Predict η*(λ, p, d) in closed form (or a tight bound).
- **Empirical test:** sweep weight decay λ ∈ {0.3, 1.0, 3.0} × noise rate η × 5 seeds. Predict η* shifts left (higher noise tolerance) as λ grows. ≈45 toy runs, ~3 hours on a single GPU.

**Cost on Rivanna:** Negligible compute (<5 GPU-hr). Months of analyst time.

**Risk assessment:**
- *Low for the empirical test, high for the theory itself.* The theory might not yield a clean closed form; the prediction might be qualitatively right but quantitatively off by 2–3×.
- Reviewer-killer: if the predicted η* is off by a factor of 2, the theory is "explanatory but not predictive" — still useful but weaker.

**Outcome envelope:**
- *Best case:* Closed-form η*(λ, p, d) matches sweep within 20%. → Headline figure: "predicted vs measured cliff." Anchors the paper as a theory contribution.
- *Likely case:* Qualitative scaling (η* ∝ √λ) confirmed. → Solid Section 5.
- *Worst case:* Sweep contradicts theory. → Theory dropped, paper is empirical-only.

---

## 4. Recommended path to a top-venue submission

The ranking depends on what role we want this paper to play.

| Goal | Best path | Realistic venue | Timeline |
|---|---|---|---|
| **Mechanistic interpretability flagship** | A + C (rescue + theory) | NeurIPS 2026 | 2.5 months |
| **Practical contamination diagnostic** | B (real-LM + 3rd-party detection) | NeurIPS 2026 | 3 months |
| **Tight self-contained workshop paper** | Polish current results, add A | NeurIPS 2026 MechInterp WS or ICLR 2027 | 1 month |

**Recommendation:** Go for **A + C now, B in parallel as a parachute.**
- A and C are *low-cost, high-confidence* — we already have all the checkpoints needed for A; C needs ≤45 toy runs and a literature-driven derivation. Together they make the *toy* claim genuinely novel ("contamination causally suppresses a specific circuit, and we predict the threshold").
- B is *high-cost, high-variance* — even if it fails, the negative result is publishable as "geometry signals do not (yet) generalize from toy to LM-scale," which is itself a useful finding.

**Critical:** **Do not submit the current results as-is.** The noise-baseline equivalence is fatal to the framing as it stands. Reviewers who read the noise-baseline numbers will reject. The paper must either (i) drop the "model-collapse vs random-noise distinction" framing entirely and pivot to "label-noise rate determines cleanup-phase success" (with the noise-collapse equivalence as a *finding*, not a buried inconsistency), or (ii) demonstrate the dissociation with experiment B on real LMs where noise-vs-synthetic-data really *should* differ.

---

## 5. Suggested paper outline (NeurIPS 2026 main track, 9 pages)

Working title: **"Contamination Cliffs: Causal Mechanism and Threshold Theory for Grokking Failure under Label Noise"**

- **§1 Introduction.** Frame: model collapse from synthetic data is a real concern (Shumailov, Dohmatob); on toy tasks, we show it is *equivalent to label noise* and admits a *causal mechanistic explanation* and a *quantitative threshold prediction*. Headline claims:
  - (C1) Sharp grokking cliff between 5% and 15% label corruption, dependent only on the *rate*, not the corruption distribution shape.
  - (C2) The cliff is *causally* explained by suppression of the Fourier circuit in MLP-out: surgical reinstatement rescues grokking.
  - (C3) A label-noise + weight-decay theory predicts the cliff position; the prediction is verified by a weight-decay sweep.
  - (C4) (Stretch — workshop appendix if B partial) Geometry-based metrics in real LMs partially track contamination but lag perplexity; we discuss why.

- **§2 Background.** Grokking (Power 2022, Liu 2022, Nanda 2023). Progress measures (Chan 2023). Model collapse (Shumailov 2024, Dohmatob 2024). Label-noise theory (Frei 2022, Bai 2021). Position our claim against each.

- **§3 The cliff and the noise–collapse equivalence.**
  - 4×3×5 grid + noise + scarcity baselines (already collected).
  - Headline figure: 2-panel — (a) test-acc vs contamination level for {noise, collapse, scarcity}, showing noise≈collapse, scarcity flat; (b) Fourier concentration for the same.
  - Statistical tests (Mann-Whitney, BH correction).

- **§4 Causal rescue (Experiment A).**
  - Mechanistic identification of the Fourier circuit.
  - Patch design: pure→contam transplant of MLP-out projection.
  - Result: rescue rate vs contamination level vs patch component.
  - Specificity control: random-direction patches do not rescue.

- **§5 Threshold theory (Experiment C).**
  - Derivation of η*(λ, p, d).
  - Weight-decay sweep predicting η* shift.
  - Why severity is irrelevant: the cleanup-phase gradient is sensitive to *any* deviation from clean, and the deviation magnitude saturates above a small severity.

- **§6 Real-LM extension (Experiment B, partial).**
  - GPT-2 medium + LoRA on contaminated OpenWebText.
  - Geometry metrics (effective rank, attention entropy, feature density) tracked.
  - Result: signals correlate with contamination but with weaker pre-grok-style leadtime than the toy. Honest discussion.
  - Public-model probe: Phi-3 vs Mistral vs Pythia ranking via geometry — report whatever we find, including null results.

- **§7 Discussion and limitations.**
  - The paper's empirical contribution is on a 1-layer toy. The mechanistic claim is causal at toy scale. The theory is tested only at toy scale.
  - The contamination ≈ noise equivalence is *a finding*, not a bug. We argue the field has been studying two names for the same phenomenon under most threat models.
  - Stretch claims (B) are flagged as preliminary.

- **§8 Conclusion.**

**Appendices:** seed-sweep details, the rescue specificity controls, the closed-form derivation, hyperparameter robustness, ablations on prime / model dim / weight decay.

---

## 6. Target venues, ordered by feasibility

| Venue | Deadline (approx.) | Notes |
|---|---|---|
| NeurIPS 2026 main track | ~22 May 2026 | **2 weeks from today.** Only feasible with dramatic scope cut (A+C as full paper, B as appendix). Still risky given baseline-equivalence weakness; reviewer pool sees label-noise-on-grokking as incremental. |
| NeurIPS 2026 MechInterp / Synthetic-Data Workshops | mid-July 2026 | **2 months from today.** Comfortable. A+C produces a strong workshop paper; B can be a separate workshop short paper if it lands. |
| ICLR 2027 | mid-Sep 2026 | **4 months from today.** Comfortable for the *full* A+B+C story. Best fit if Experiment B partially lands. |
| ICML 2027 main | end-Jan 2027 | **8 months.** If B is a hit, this is the venue with the largest mechanistic/synthetic-data audience. |

**Recommended:** Aim for **NeurIPS workshop in July** as the safe landing for A+C, simultaneously running B at production scale through July–August, then combine the strongest version of all three for **ICLR 2027 (Sep deadline)**.

---

## 7. What to do this week

1. **Confirm or refute Experiment A's feasibility on existing checkpoints.**
   We have 10 checkpoints per condition × 5 seeds × 12 grid cells. Pick `pure/seed_42/checkpoint_50000.pt` and `level0.15_sev0.6/seed_42/checkpoint_50000.pt`; do a 1-day mechanistic sandbox (Nanda-style Fourier basis identification) and try a single MLP-out patch. If even the first attempt shows a signal, A is on. If it shows nothing, A is much harder than the writeup above suggests.
2. **Start the weight-decay sweep for Experiment C.** Tiny runs, can be queued today. λ ∈ {0.3, 1.0, 3.0} × η ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.30} × 5 seeds = 90 runs ≈ 1 GPU-day total. The result is independent of A and feeds the theory section directly.
3. **Let `real-gen` finish, then queue the 15-run real-LM training array for B.** Per-run cost is fixed; the sooner we start, the more buffer for revision cycles.
4. **Frame-shift the writeup.** Drop "model-collapse vs noise" as the central distinction. Reframe around "label-noise rate as the causal driver of grokking cleanup-phase failure, with noise-collapse equivalence as a key empirical finding."

---

## 8. The brutal one-paragraph answer

The current results are a competent, well-executed *toy* study whose central novelty claim — that "model-collapse contamination" is a distinct phenomenon from "label noise" — is **disproved by our own noise baseline**. Without further work, this is a workshop paper at best and arguably a negative result. To reach a top venue, we need (A) a *causal* rescue experiment that elevates the Fourier-circuit story from correlation to causation, (C) a *theoretical* derivation of the cliff threshold that explains both the level-only effect and the noise-collapse equivalence, and ideally (B) a real-LM extension showing geometry-based metrics generalize. A and C are achievable in 1–2 months at low compute cost; B is the high-variance, high-reward parallel track that determines whether this is a strong NeurIPS submission or an ICLR submission with a workshop fallback. **Do not submit the current results to a main track without at least A and C.**
