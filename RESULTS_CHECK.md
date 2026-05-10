# Results check — 2026-05-10

Cross-project verification of new experimental results that landed today.
Numbers below are from raw on-disk JSONs/logs, not summaries.

## 1. Grokking — circuit transplant rescue (this repo)

Source: `analysis/transplant/wd1_n{0.15,0.2}_s{42..46}/rescue_results.json`
(N = 5 seeds × 2 noise levels = 10 runs, completed by job 12740470_[0..9]).

Pure baseline: `wd=1.0, noise=0, train_frac=0.3`, eval after 50k steps. Pure
test-acc mean ≈ 0.984 (one seed grokked at 0.918, four at 1.000).

### test_acc, mean ± std (n=5 per noise level)

| component | noise | transplant   | transplant+rt    | rand          | rand+rt       |
|-----------|-------|--------------|------------------|---------------|---------------|
| token_embed       | 0.15 | 0.017±0.002 | **0.511±0.226** | 0.016±0.002 |   0.003±0.000 |
| token_embed       | 0.20 | 0.018±0.003 | **0.464±0.115** | 0.016±0.001 |   0.006±0.003 |
| self_attn_in_proj | 0.15 | 0.019±0.003 |   0.782±0.022   | 0.019±0.004 |   0.663±0.337 |
| self_attn_in_proj | 0.20 | 0.018±0.006 |   0.719±0.038   | 0.017±0.002 |   0.737±0.029 |
| self_attn_out_proj| 0.15 | 0.017±0.002 |   0.791±0.033   | 0.019±0.002 |   0.790±0.029 |
| self_attn_out_proj| 0.20 | 0.017±0.005 |   0.705±0.035   | 0.015±0.001 |   0.717±0.039 |
| linear1           | 0.15 | 0.019±0.002 | **0.294±0.348** | 0.020±0.002 |   0.028±0.024 |
| linear1           | 0.20 | 0.018±0.001 |   0.126±0.204   | 0.017±0.000 |   0.011±0.003 |
| linear2           | 0.15 | 0.018±0.001 |   0.768±0.043   | 0.018±0.002 |   0.794±0.024 |
| linear2           | 0.20 | 0.019±0.003 |   0.473±0.264   | 0.017±0.001 |   0.486±0.270 |
| output_head       | 0.15 | 0.019±0.011 | **0.776±0.070** | 0.016±0.006 |   0.116±0.010 |
| output_head       | 0.20 | 0.023±0.007 | **0.646±0.146** | 0.015±0.003 |   0.121±0.024 |

(`+rt` = freeze the swapped component, retrain everything else for 2000 steps.
"Specific" = pure-transplant+rt rescues but rand+rt does NOT.)

Verdicts on the prior claims:

- **Is `token_embed` really the specific circuit? — YES.** transplant+rt
  rescues to 0.46–0.51 acc, rand+rt stays at 0.003–0.006 (worse than the
  original contam baseline ~0.84). Gap is huge (>0.45) and consistent
  across all 10 seeds. **Confirmed.**
- **Is `linear2` really NOT specific? — YES.** transplant+rt and rand+rt
  produce indistinguishable accuracies (0.768 vs 0.794 at n0.15; 0.473
  vs 0.486 at n0.20). **Confirmed.**

Other components (worth flagging for the writeup, not asked but visible in
the data):

- `output_head`: ALSO specific (0.78 transplant+rt vs 0.12 rand+rt at n0.15).
- `self_attn_{in,out}_proj`: NOT specific (transplant+rt ≈ rand+rt).
- `linear1`: appears specific BUT very high variance (0.348 stddev at
  n0.15) — 2/5 seeds rescue, 3/5 fail. Need more seeds before claiming this.
- Note: `swap_all` test-acc identical to `baseline_pure` (0.984) — the
  variant just evaluates the unmodified pure model, not a contamination
  test. Working as intended but possibly mislabeled in the table.

## 2. Grokking — prime sweep

Source: `results/prime_sweep/p{59,97,113}/wd1/noise*/seed_*/results.json`
(target: 5 seeds × {0, 0.05, 0.1, 0.15, 0.2, 0.3} per prime = 30 cells/prime).

### final test_acc mean (n=5 per cell)

| noise → | 0    | 0.05  | 0.1   | 0.15  | 0.2   | 0.3   |
|---------|------|-------|-------|-------|-------|-------|
| **p=59**  | 0.984 | 0.977 | 0.914 | 0.827 | 0.752 | 0.248 |
| **p=97**  | 1.000 | 0.989 | 0.972 | 0.884 | 0.754 | 0.518 |
| **p=113** | 1.000 | 0.984 | 0.970 | 0.883 (n=3) | — | — |

Status of cells:
- p=59: complete (this is the original wd1 grid; 30/30 cells).
- p=97: complete (30/30 cells).
- p=113: 23/30 cells. noise=0.15 has only 3/5 seeds; noise=0.2 and
  noise=0.3 have 5 seed dirs each but no `results.json` yet (only
  intermediate checkpoints). Job array 12740472 still has tasks
  84–89 running per `squeue` snapshot; the tail of the array is the
  large-prime cells (p=113 takes longer per training step).

### Does the cliff replicate across primes?

**Partially — and the location shifts.** A monotonic decline with
noise is present for both p=59 and p=97. The "cliff" between
noise=0.2 and noise=0.3 is sharp for p=59 (0.752 → 0.248, drop 0.504)
but only moderate for p=97 (0.754 → 0.518, drop 0.236). p=113 cannot
be compared past noise=0.15 yet — **claims about "the cliff replicates
at p=113" are premature** until 12740472 finishes.

The p=59 cliff is clearly steeper than p=97 — the qualitative claim
"there is a cliff" replicates at p=97, but not in the sharp form seen
at p=59. Need p=113 cells at noise≥0.2 before any prime-scaling story.

## 3. Grokking — real-gen v2 (`data/contaminated_real/`)

Source: directory listing + `dataset_info.json`.

Contents on disk:
- `clean_train/`           1 .arrow shard set, mtime 2026-05-10 14:13
- `test/`                  1 .arrow shard, mtime 2026-05-10 14:13
- `ratio_0/seed_{0,1,2}/`  3 seeds, mtimes ~14:13
- `ratio_5/seed_{0,1}/`    2 seeds (seed_2 missing), mtimes ~23:19, 08:19
- `ratio_{10,15,20}/`      **NOT created**

`config.json` declares `ratios_pct=[15.0]` and `seeds=[2]` — i.e., the
generator script writes one cell per invocation and the file reflects
the most-recent target. Job 12740481 (grok-real-gen array) is currently
running tasks 5,6,7,8 with 9–17 pending (`JobArrayTaskLimit`), so the
remaining ratio×seed cells are still being filled.

**Populated cells: ratio_0 × {0,1,2} and ratio_5 × {0,1}. Everything
else (ratio_5/seed_2, ratio_10/*, ratio_15/*, ratio_20/*) is missing.**
No claim about cross-ratio comparison can be made yet from this dataset.

## 4. ESM2 — ablation diagnostic (job 12740474)

Source: `/scratch/qzp4ta/esm2-sae-diffusion/logs/esm2sae-ablate-diag-12740474.{out,err}`,
`/scratch/qzp4ta/esm2-sae-diffusion/results/ablation_diagnostic/diagnostic.json`.

Job COMPLETED (33 s on A100). Tested 5 features {76370, 46632, 159682,
37277, 157758} on 3 pilot proteins (GB1 L=56, TrpCage L=20, Villin L=35)
across 4 ablation scales {1.0, 2.0, 5.0, 10.0}.

**Finding (verbatim from the JSON, every cell):**
- `mean_z = 0.0`, `max_z = 0.0`, `n_active_residues = 0`,
  `active_rate = 0.0` for all 5 features × all 3 proteins
- `delta_z_mean = delta_z_min = delta_z_max = 0.0` at every scale
- `n_residues_killed_to_zero = 0` (because they were never above zero)
- `n_residues_now_zero_total = L` (whole protein)

**Diagnosis:** the no-op ablation result is explanation (a) from the
diagnostic's framing — these features are completely silent on the pilot
proteins (z = 0 everywhere). Ablating zero gives zero delta. So the
ablation pipeline is not buggy; the SAE features themselves don't fire
on the test substrates. Per-feature thresholds were 0.10 and `max_z`
never reached even 0.0, so the gap (`max_z_minus_threshold = -0.10`)
isn't even close.

## 5. ESM2 — DSSP run

Source: `logs/esm2sae-dssp-{12740473,12741354,12741569}.{out,err}` and
`results/dssp_annotations.{json,npz}`.

Status (per `sacct -X`):
- 12740473 FAILED at 18:54 (5 s) — `pip: command not found` in sbatch script.
- 12741354 FAILED at 19:11 (1 s) — same kind of setup error.
- **12741569 COMPLETED at 19:37 (2:37)** — successful end-to-end mdtraj run.

So **yes, the mdtraj-based run started — and finished**. But the result
matters more than completion:

From `results/dssp_annotations.json`:
- `n_chains_used = 183`, `n_residues = 30,492`
- Q3 prevalence sane: H=0.389, E=0.217, C=0.394 (sums to 1.0)
- **`nnz = 1` across the entire 30,492 × 163,840 SAE activation matrix
  (density ≈ 2e-10).** The single nonzero is on PDB 1yo7 chain A.
- Top features per H/E/C class: feature `47831` for all three, AUROC =
  0.500 (random) for all three.

The `chain_index` confirms this: every other entry has `n_nonzero = 0`,
only `1yo7` has `n_nonzero = 1`.

**This is consistent with finding #4 (the SAE simply doesn't fire on real
ESM2 activations) and supersedes it — it's a near-total dead-feature
situation, not specific to the 5 ablation-pilot features. The DSSP results
file exists but contains essentially no signal.**

## 6. UVAVAE — recon quality (job 12740476, COMPLETED 2:31)

Source: `/scratch/qzp4ta/uvavae/results/recon_quality/{plain_vae,local_only,global_cka,spatial_relational,hybrid}.json`
plus `summary.json`. n=5,000 ImageNet val images per variant.

| variant            | PSNR↑   | SSIM↑   | L1↓      | LPIPS↓  |
|--------------------|---------|---------|----------|---------|
| **plain_vae**      | **32.092** | **0.7627** | **0.03629** | **0.1649** |
| local_only         | 31.412  | 0.7455  | 0.03924  | 0.1771  |
| global_cka         | 31.410  | 0.7454  | 0.03927  | 0.1769  |
| spatial_relational | 31.445  | 0.7467  | 0.03908  | 0.1755  |
| hybrid             | 31.412  | 0.7458  | 0.03926  | 0.1759  |

**Is plain_vae really better at reconstruction? — YES, on every metric.**
Gap is uniform: ≈0.68 dB PSNR, +0.017 SSIM, −0.003 L1, −0.012 LPIPS.
Among the four "enhanced" variants, the four are nearly indistinguishable
(within 0.04 dB PSNR / 0.001 SSIM of each other).

### Cross-check vs DiT downstream metrics

`results/in100_fixed_per_variant/*.json` (DiT-generated samples from each
tokenizer, IN-100):

| variant            | rFID↓   | IS↑     | Precision↑ | Recall↑ |
|--------------------|---------|---------|------------|---------|
| **plain_vae**      | **29.339** | **47.05** | **0.591**  | 0.369   |
| local_only         | 29.769  | 41.10   | 0.434      | 0.406   |
| global_cka         | 29.634  | 39.64   | 0.439      | 0.424   |
| spatial_relational | 29.693  | 36.19   | 0.371      | 0.391   |
| hybrid             | 29.917  | 37.60   | 0.385      | 0.413   |

Plain_vae also wins rFID, IS, and precision; loses on recall (diversity).

DiT training loss (smoothed last-50 train, lowest val), step 79170,
from `dit_imagenet100_*/logs/metrics.jsonl`:

| variant            | train_loss(smooth) | val_loss(last) | val_loss(min) |
|--------------------|---------------------|-----------------|----------------|
| plain_vae          | 0.247               | 0.239           | 0.224          |
| local_only         | 0.100               | 0.092           | 0.086          |
| global_cka         | 0.100               | 0.092           | 0.086          |
| spatial_relational | 0.099               | 0.091           | 0.086          |
| hybrid             | 0.100               | 0.092           | 0.086          |

**Story consistency — partially consistent, with a real DiT-loss inversion:**

- Recon quality and rFID/IS agree: plain_vae best, others tied for worse.
- DiT loss disagrees: plain_vae is 2.5× higher than the others, yet
  produces the best generations. Whatever lowers DiT loss for the
  enhanced variants (likely smaller-magnitude or more Gaussian latents)
  doesn't translate to better samples; the enhanced variants train an
  "easier" diffusion target without any downstream win.
- Caveat: DiT loss is in latent units that depend on each tokenizer's
  latent scale, so cross-variant loss comparison is not strictly
  apples-to-apples — but the 2.5× gap is large enough that the qualitative
  inversion stands.

## 7. DiffGuard — DRS ablation grid (job 12740479, RUNNING)

Source: `/scratch/qzp4ta/diffusion-injection-detect/results/drs_ablation_12740479/*/cell_metrics.json`
plus log `logs/dg-drs-ablation-12740479.out`.

Grid: 2 datasets × 2 base models × 5 finetune-step counts = **20 cells**.
Job has been running 26 min.

**Cells with `cell_metrics.json` written so far: 4 / 20.**
A 5th cell (sst2 / distilbert / 10000 steps) is currently mid-finetune
(step 9900/10000 in tail of log). All completed cells are sst2 +
distilbert; nothing for sst2 + bert-base, ag_news + distilbert, or
ag_news + bert-base yet.

| dataset | model | steps | weighted | best_single | loss_baseline | Δ vs best | Δ vs baseline |
|---------|-------|-------|----------|-------------|---------------|-----------|----------------|
| sst2 | distilbert | 500   | 0.479 | 0.536 | 0.507 | −0.057 | −0.028 |
| sst2 | distilbert | 1500  | 0.490 | 0.548 | 0.525 | −0.058 | −0.035 |
| sst2 | distilbert | 3000  | 0.500 | 0.561 | 0.539 | −0.061 | −0.039 |
| sst2 | distilbert | 5000  | 0.514 | 0.577 | 0.557 | −0.063 | −0.043 |

Observations on the early cells:
- The **weighted-LR composite is consistently worse than the loss
  baseline** by 0.028–0.043 AUROC, and worse than the best-single-mask
  signal by 0.057–0.063. Negative gains across all 4 cells so far.
- Detection signal does grow with more finetune steps for the loss
  baseline (0.507 → 0.557) and best-single (0.536 → 0.577), confirming
  more memorization is more detectable. The weighted detector tracks
  this trend but never catches up.
- weighted_lr_auroc_train (0.563 at 500 steps) >> weighted_lr_auroc_test
  (0.479 at 500 steps) → train/test gap is ~0.08, suggesting the
  weighted-LR is overfitting the per-mask features.

**Cannot say anything about ag_news or bert-base yet — those 16 cells
haven't started.**

## Summary of discrepancies and watchpoints

1. **ESM2 SAE looks effectively dead on real proteins.** Both the
   targeted ablation diagnostic (5 pilot features × 3 proteins) and
   the broad DSSP run (163,840 features × 30,492 residues = 5e9
   activations, only **1** nonzero) point to the same root cause. Any
   prior claim about "feature 76370 codes for X" or AUROCs from this
   SAE checkpoint should be re-checked against the dead-feature reality.
2. **UVAVAE DiT loss is anti-correlated with downstream rFID** — the
   enhanced variants train ~2.5× lower DiT loss but produce slightly
   worse samples than plain_vae. If anyone has previously argued that
   lower DiT loss = better tokenizer in this codebase, that claim is
   contradicted by these new numbers.
3. **Grokking prime sweep at p=113 is incomplete** — noise=0.15 has 3/5
   seeds, noise={0.2, 0.3} have 0/5. Cliff replication claim for p=113
   is premature; reassess once 12740472 finishes.
4. **Grokking real-gen v2 has only 5/N cells filled** — anything past
   ratio_5 is unavailable. Job array 12740481 limited by
   `JobArrayTaskLimit`, so this will fill in over time.
5. **DiffGuard DRS ablation: 4/20 cells.** Early signal: weighted
   detector loses to loss baseline by 0.03–0.04 AUROC across all
   completed sst2/distilbert cells. Wait for the cross-model and
   cross-dataset cells before drawing conclusions.

Verifications of prior asks:
- transplant: `token_embed` IS specific (✓), `linear2` is NOT specific (✓).
- prime sweep: cliff is real for p=59 and p=97; sharper for p=59; p=113 TBD.
- ESM2 ablation diagnostic: explanation (a) — z=0 already; ablation no-op
  is a downstream symptom of dead features.
- DSSP: mdtraj-based run completed and wrote `dssp_annotations.json`,
  but output is essentially empty (1 nonzero / 5e9).
- UVAVAE recon: plain_vae IS better at reconstruction (✓).
- DiffGuard: ablation is running but only 4/20 cells in.
