"""
Experiment C — threshold theory for the grokking cliff.

Two outputs:

  1. A short, explicit derivation (text printed at the top of the script and
     re-emitted into analysis/threshold_theory.md) that predicts a cliff
     position η*(λ, p, d) up to constants.

  2. An empirical fit against the wd × noise grid in `results/exp_c_grid/`,
     which extracts per-(wd, seed) η* from the grokked/not-grokked observations
     and tests a power-law scaling η* = c · λ^b vs. the predicted η* ∝ λ.

DERIVATION SKETCH
-----------------
At the memorization fixed point θ_mem, AdamW's per-step drift is approximately

    d θ ≈ -lr · ( ∇L(θ; data) + λ θ )

where the data-loss gradient on training batches contains two components:

    ∇L = (1 - η) · ∇L_clean + η · ∇L_noise

with η the rate of label-corruption. At θ_mem the *clean* gradient is small
(model has memorized clean examples), but the *noise* gradient is bounded
below: each corrupted example pushes the logits in a wrong direction with
norm O(1) per example. Concretely,

    || ∇L_noise(θ_mem) ||  =  Θ(1)              [bounded below away from 0]
    || ∇L_clean(θ_mem) ||  =  o(1)              [clean loss is at its min]

Cleanup proceeds along the weight-decay flow d θ = -λ θ. Noise opposes this
by maintaining a stochastic gradient floor of order η. The cleanup phase makes
*net* progress toward smaller-norm structured solutions iff the decay drift
dominates the noise drift:

    λ · || θ_mem ||  >  η · || ∇L_noise(θ_mem) ||                   (★)

⇒  η  <  λ · || θ_mem || / || ∇L_noise ||

For the modular-arithmetic transformer, || θ_mem || at memorization scales
with √(p · d_model) (each of the p tokens learns a d-dimensional embedding,
plus FFN matrices contribute the same order in any structured memorization
solution). The per-example logit gradient is O(1). Therefore

    η*(λ, p, d)  ≈  C · λ · √(p · d_model)                          (P1)

for some constant C absorbing the lr / batch / Adam preconditioner.

REGIME II — too-much-decay catastrophe
--------------------------------------
The argument above implicitly assumes θ_mem is itself a stable fixed point
under the combined gradient + weight-decay flow. That stability requires

    || ∇L_clean(θ_mem) ||  >  λ · || θ_mem ||                       (★★)

(otherwise weight decay alone shrinks θ below the memorization point even on
clean data). When (★★) is *violated* — i.e. λ is so large the model cannot
even memorize the clean training set — there is no cleanup phase to discuss.
Empirically this corresponds to wd = 3.0 in our grid: train_acc never reaches
100%, the run fails for *all* noise levels (including noise=0). The model
operates in a different regime entirely.

EMPIRICAL TESTS THIS SCRIPT RUNS
--------------------------------
- Per (wd, seed), define η*(wd, seed) as the smallest noise level at which
  the run did NOT reach test_acc ≥ 0.95. If all noise levels grokked, treat
  η* as right-censored at the largest tested noise. If none grokked (regime
  II), treat as 0.
- Fit log η*(λ) = log C + b · log λ via OLS on the median η* per λ.
  Prediction (P1) says b = 1; constants C absorb p,d which are fixed.
- Bootstrap a 95% CI around b across seeds.
- Save analysis/threshold_theory_summary.md and a fit plot.

This is a 1-day analyst pass over the grid we already have. The closed-form
prediction is a hypothesis; the script will print whether the data supports
b ≈ 1, contradicts it, or is too low-resolution to tell.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROK_THRESHOLD = 0.95


def parse_grid(grid_dir: Path) -> List[dict]:
    """Walk results/exp_c_grid/wd*/noise*/seed_*/results.json."""
    rows: List[dict] = []
    for wd_dir in sorted(grid_dir.glob("wd*")):
        m = re.match(r"^wd([\d.]+)$", wd_dir.name)
        if not m:
            continue
        wd = float(m.group(1))
        for noise_dir in sorted(wd_dir.glob("noise*")):
            m2 = re.match(r"^noise([\d.]+)$", noise_dir.name)
            if not m2:
                continue
            noise = float(m2.group(1))
            for seed_dir in sorted(noise_dir.glob("seed_*")):
                ms = re.match(r"^seed_(\d+)$", seed_dir.name)
                if not ms:
                    continue
                seed = int(ms.group(1))
                res_path = seed_dir / "results.json"
                if not res_path.exists():
                    continue
                with res_path.open() as f:
                    data = json.load(f)
                rows.append({
                    "wd": wd,
                    "noise": noise,
                    "seed": seed,
                    "grokked": bool(data.get("grokked", False)),
                    "final_test_acc": float(data.get("final_test_acc", 0.0)),
                    "final_train_acc": float(data.get("final_train_acc", 0.0)),
                    "final_fourier": float(data.get("final_fourier_concentration", 0.0)),
                })
    return rows


def per_seed_threshold(rows: List[dict]) -> Dict[Tuple[float, int], float]:
    """For each (wd, seed) return η* = smallest noise at which the run failed
    (test_acc < threshold). If all noises grokked, set η* = max+ε (right-cens).
    If none grokked, set η* = 0 (left-cens / regime II)."""
    by_key: Dict[Tuple[float, int], List[Tuple[float, bool]]] = {}
    for r in rows:
        by_key.setdefault((r["wd"], r["seed"]), []).append(
            (r["noise"], r["final_test_acc"] >= GROK_THRESHOLD))
    out: Dict[Tuple[float, int], float] = {}
    for k, lst in by_key.items():
        lst.sort(key=lambda t: t[0])
        noises = [n for n, _ in lst]
        groks = [g for _, g in lst]
        max_noise = max(noises) if noises else 0.0
        if not any(groks):
            out[k] = 0.0
            continue
        # Find smallest noise where grok=False, that is ≥ smallest noise where grok=True.
        first_fail = None
        for n, g in lst:
            if not g:
                first_fail = n
                break
        if first_fail is None:
            out[k] = max_noise + 0.01
        else:
            out[k] = first_fail
    return out


def fit_power_law(wds: np.ndarray, etas: np.ndarray) -> Tuple[float, float, float]:
    """Fit log η* = log C + b · log λ via OLS. Returns (C, b, R^2)."""
    mask = (etas > 0) & (wds > 0) & np.isfinite(etas) & np.isfinite(wds)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    x = np.log(wds[mask])
    y = np.log(etas[mask])
    A = np.column_stack([np.ones_like(x), x])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    log_c, b = sol
    yhat = A @ sol
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() if y.size > 1 else 0.0
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(np.exp(log_c)), float(b), float(r2)


def bootstrap_ci(per_seed: Dict[Tuple[float, int], float],
                 n_boot: int = 1000, alpha: float = 0.05,
                 rng_seed: int = 0) -> Tuple[float, float]:
    """Bootstrap CI on the exponent b across seeds."""
    rng = np.random.RandomState(rng_seed)
    seeds = sorted({s for _, s in per_seed.keys()})
    wds = sorted({w for w, _ in per_seed.keys()})
    bs: List[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        med_etas = []
        for w in wds:
            es = [per_seed[(w, s)] for s in sampled if (w, s) in per_seed]
            es = [e for e in es if e > 0]
            med_etas.append(float(np.median(es)) if es else float("nan"))
        med = np.array(med_etas)
        wd_arr = np.array(wds)
        _, b, _ = fit_power_law(wd_arr, med)
        if math.isfinite(b):
            bs.append(b)
    if not bs:
        return float("nan"), float("nan")
    bs.sort()
    lo = bs[int((alpha / 2) * len(bs))]
    hi = bs[int((1 - alpha / 2) * len(bs))]
    return lo, hi


def write_summary(out_path: Path, rows, per_seed, fit, ci, regime_ii_wds):
    C, b, r2 = fit
    lo, hi = ci
    grid_wds = sorted({r["wd"] for r in rows})
    grid_noises = sorted({r["noise"] for r in rows})
    n_seeds = len({r["seed"] for r in rows})

    lines = ["# Threshold Theory — Empirical Fit (Experiment C)\n\n"]
    lines.append(
        "## Theoretical prediction\n\n"
        "From the cleanup-phase balance condition (★) in `src/threshold_theory.py`:\n\n"
        "$$\\eta^*(\\lambda, p, d)  \\;\\approx\\;  C \\cdot \\lambda \\cdot \\sqrt{p \\cdot d_{\\text{model}}}$$\n\n"
        "Holding p and d fixed (this study: p=59, d=128), the prediction reduces to "
        "**η* ∝ λ¹** (i.e. exponent b = 1 in η* = C·λ^b), valid only in the regime where (★★) holds — "
        "i.e. weight decay does not by itself destabilize the memorization solution.\n\n"
    )
    lines.append("## Setup\n\n")
    lines.append(
        f"- Grid: wd ∈ {grid_wds}, noise ∈ {grid_noises}, n_seeds = {n_seeds}\n"
        f"- η*(λ, seed) := smallest noise level where the run failed to reach test_acc ≥ {GROK_THRESHOLD}.\n"
        f"- If all noise levels grokked → right-censored at max(noise) + 0.01.\n"
        f"- If no noise level grokked (regime II / decay too large) → η* = 0 and excluded from the fit.\n\n"
    )
    if regime_ii_wds:
        lines.append(
            f"## Regime II detected at wd ∈ {regime_ii_wds}\n\n"
            "These wd values prevent grokking even at noise=0. Per the derivation, this corresponds to "
            "violation of the memorization-stability condition (★★): λ·||θ_mem|| > ||∇L_clean(θ_mem)||. "
            "These points are excluded from the cliff-shift fit because there is no cliff to fit — the "
            "model fails everywhere.\n\n"
        )
    lines.append("## Per-(wd, seed) cliff position\n\n")
    lines.append("| wd | seed | η* |\n|---|---|---|\n")
    for (wd, seed), eta in sorted(per_seed.items()):
        lines.append(f"| {wd} | {seed} | {eta:.4f} |\n")
    lines.append("\n## Power-law fit\n\n")
    lines.append(
        f"- Fitted η* = **{C:.4f} · λ^{b:+.3f}**\n"
        f"- R² = {r2:.3f}\n"
        f"- Bootstrap 95% CI on b: [{lo:.3f}, {hi:.3f}]\n"
        f"- Theory predicts b = 1.0; empirical b = {b:+.3f}.\n\n"
    )
    if math.isfinite(b) and lo <= 1.0 <= hi:
        lines.append("**Verdict:** the predicted exponent b=1 lies inside the 95% CI. "
                     "Theory is consistent with the empirical scaling at this resolution.\n\n")
    elif math.isfinite(b):
        lines.append(
            f"**Verdict:** predicted b=1 is *outside* the 95% CI [{lo:.3f}, {hi:.3f}]. "
            "Either the constant assumptions in the derivation are wrong (e.g. ||∇L_noise|| also scales "
            "with λ via Adam preconditioning), or the discretisation of η in the grid is too coarse to "
            "resolve the cliff shift. Recommend: rerun a finer noise sweep at η ∈ {0.06, 0.08, 0.10, "
            "0.12, 0.14, 0.16} for two wd values.\n\n"
        )
    else:
        lines.append(
            "**Verdict:** insufficient non-censored cells to fit. Rerun with finer noise resolution "
            "or report a *bound* (η* > 0.10 at wd ∈ {0.3, 1.0}) rather than a fit.\n\n"
        )
    lines.append("## How to reproduce\n\n")
    lines.append("```bash\n"
                 "python src/threshold_theory.py --grid-dir results/exp_c_grid --output-dir analysis/\n"
                 "```\n")
    out_path.write_text("".join(lines))


def plot_fit(per_seed, fit, regime_ii_wds, out_path):
    C, b, r2 = fit
    pts = [(wd, eta) for (wd, _), eta in per_seed.items() if eta > 0]
    if not pts:
        return
    wds_p = np.array([p[0] for p in pts], dtype=float)
    etas_p = np.array([p[1] for p in pts], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(wds_p, etas_p, alpha=0.5, label="per-seed η*", color="#1f77b4")
    # Median per wd
    wds_unique = sorted(set(wds_p.tolist()))
    medians = [np.median(etas_p[wds_p == w]) for w in wds_unique]
    ax.plot(wds_unique, medians, marker="o", color="#d62728", label="median η* per wd")
    if math.isfinite(C) and math.isfinite(b):
        xs = np.geomspace(min(wds_unique) * 0.8, max(wds_unique) * 1.2, 50)
        ax.plot(xs, C * xs ** b, color="#2ca02c", linestyle="--",
                label=f"fit: η* = {C:.3f}·λ^{b:+.2f}")
        # Theory line: η* = constant * λ^1, anchored at the median wd
        wd_anchor = wds_unique[len(wds_unique) // 2]
        eta_anchor = medians[len(medians) // 2]
        ax.plot(xs, eta_anchor / wd_anchor * xs, color="black", linestyle=":",
                label="theory: η* ∝ λ¹")
    for wd in regime_ii_wds:
        ax.axvline(wd, color="grey", alpha=0.3, linestyle=":")
        ax.text(wd, 0.02, f"regime II", rotation=90, alpha=0.6, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("weight decay λ")
    ax.set_ylabel("cliff η*")
    ax.set_title("Empirical η*(λ) vs theory")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", type=Path,
                    default=Path("results/exp_c_grid"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("analysis"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_grid(args.grid_dir)
    if not rows:
        raise SystemExit(f"no results found under {args.grid_dir}")
    print(f"[info] loaded {len(rows)} runs across "
          f"{len({(r['wd'], r['noise']) for r in rows})} cells")

    per_seed = per_seed_threshold(rows)
    # Detect regime II (every noise failed at this wd, every seed)
    regime_ii_wds = sorted({wd for (wd, _), e in per_seed.items() if e == 0.0})
    # Build λ array from the seed×wd map (excluding regime II)
    pairs = [(w, e) for (w, _), e in per_seed.items()
             if e > 0 and w not in regime_ii_wds]
    if not pairs:
        print("[warn] no usable (wd, η*) cells; only regime II detected.")
        write_summary(args.output_dir / "threshold_theory_summary.md",
                      rows, per_seed, (float("nan"), float("nan"), float("nan")),
                      (float("nan"), float("nan")), regime_ii_wds)
        return
    wd_arr = np.array([p[0] for p in pairs])
    eta_arr = np.array([p[1] for p in pairs])
    fit = fit_power_law(wd_arr, eta_arr)
    ci = bootstrap_ci({k: v for k, v in per_seed.items()
                       if v > 0 and k[0] not in regime_ii_wds})
    print(f"[fit] η* = {fit[0]:.4f} · λ^{fit[1]:+.3f}, R²={fit[2]:.3f}")
    print(f"[boot] 95% CI on exponent b: [{ci[0]:.3f}, {ci[1]:.3f}]")

    write_summary(args.output_dir / "threshold_theory_summary.md",
                  rows, per_seed, fit, ci, regime_ii_wds)
    plot_fit(per_seed, fit, regime_ii_wds,
             args.output_dir / "threshold_theory_fit.png")
    print(f"[done] wrote {args.output_dir/'threshold_theory_summary.md'}")
    print(f"[done] wrote {args.output_dir/'threshold_theory_fit.png'}")


if __name__ == "__main__":
    main()
