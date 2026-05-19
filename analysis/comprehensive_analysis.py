"""
Comprehensive analysis combining the toy modular-arithmetic exp_c_grid (90 runs)
with the real-data contamination training (currently 4 runs: ratios 0 and 10%).

This is the report writer. Reads:
  - results/exp_c_grid/wd<W>/noise<N>/seed_<S>/results.json
  - results/contamination/ratio_<R>_seed_<S>.json
and produces:
  - analysis/comprehensive_summary.md
  - analysis/comprehensive_summary.csv         (one row per signal)
  - analysis/comprehensive_overview.png        (stitched plot)

Two strands of evidence:

  Toy (modular arithmetic + label-noise):
      mechanism = grokking, monitored by test_acc, Fourier concentration,
      embedding rank, weight norm. Outcome = grokking yes/no.

  Real (GPT-2 medium + AI-contaminated OpenWebText):
      mechanism = next-token modeling, monitored by perplexity, attention
      effective rank, repr_entropy, distinct-n. Outcome = degraded LM.

The script makes them comparable by:
  - reporting per-experiment summary tables
  - computing Spearman across both for the metrics that exist in both
    (effective rank, weight norm proxies)
  - listing what's missing on the real side and what to look at next
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_ROOT = PROJECT_ROOT / "results" / "exp_c_grid"
CONTAM_ROOT = PROJECT_ROOT / "results" / "contamination"
OUT_DIR = PROJECT_ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WD_RE = re.compile(r"^wd(?P<wd>[\d.]+)$")
NOISE_RE = re.compile(r"^noise(?P<noise>[\d.]+)$")
SEED_RE = re.compile(r"^seed_(?P<seed>\d+)$")
CONTAM_FNAME_RE = re.compile(r"^ratio_(\d+)_seed_(\d+)\.json$")


def safe_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def load_grid_runs() -> List[Dict]:
    rows = []
    if not GRID_ROOT.exists():
        return rows
    for wd_dir in sorted(GRID_ROOT.iterdir()):
        m = WD_RE.match(wd_dir.name)
        if not m or not wd_dir.is_dir():
            continue
        wd = float(m.group("wd"))
        for noise_dir in sorted(wd_dir.iterdir()):
            mn = NOISE_RE.match(noise_dir.name)
            if not mn or not noise_dir.is_dir():
                continue
            noise = float(mn.group("noise"))
            for seed_dir in sorted(noise_dir.iterdir()):
                ms = SEED_RE.match(seed_dir.name)
                if not ms or not seed_dir.is_dir():
                    continue
                seed = int(ms.group("seed"))
                p = seed_dir / "results.json"
                if not p.exists():
                    continue
                with p.open() as f:
                    data = json.load(f)
                rows.append({
                    "wd": wd, "noise": noise, "seed": seed,
                    "grokked": bool(data.get("grokked", False)),
                    "grokking_step": data.get("grokking_step"),
                    "final_test_acc": safe_float(data.get("final_test_acc")),
                    "final_train_acc": safe_float(data.get("final_train_acc")),
                    "final_fourier": safe_float(data.get("final_fourier_concentration")),
                    "final_rank": safe_float(data.get("final_embedding_rank")),
                    "final_weight_norm": safe_float(data.get("final_weight_norm")),
                })
    return rows


def load_contam_runs() -> List[Dict]:
    rows = []
    if not CONTAM_ROOT.exists():
        return rows
    for p in sorted(CONTAM_ROOT.glob("ratio_*_seed_*.json")):
        m = CONTAM_FNAME_RE.match(p.name)
        if not m:
            continue
        ratio = int(m.group(1))
        seed = int(m.group(2))
        data = json.loads(p.read_text())
        history = data.get("history", []) or []
        if not history:
            continue
        final = history[-1]
        rows.append({
            "ratio_pct": ratio,
            "seed": seed,
            "weight_decay": data.get("weight_decay"),
            "perplexity": safe_float(final.get("perplexity")),
            "train_loss": safe_float(final.get("train_loss")),
            "attn_effective_rank": safe_float(final.get("attn_effective_rank")),
            "repr_entropy": safe_float(final.get("repr_entropy")),
            "cos_sim_mean": safe_float(final.get("cos_sim_mean")),
            "distinct_2": safe_float(final.get("distinct_2")),
            "distinct_3": safe_float(final.get("distinct_3")),
            "distinct_4": safe_float(final.get("distinct_4")),
        })
    return rows


def spearman(xs, ys):
    if scipy_stats is None or len(xs) < 3:
        return None, None
    res = scipy_stats.spearmanr(xs, ys)
    rho = float(res.correlation) if hasattr(res, "correlation") else float(res.statistic)
    return rho, float(res.pvalue)


def grid_summary_block(rows):
    if not rows:
        return ["No grid runs found.\n"]
    wds = sorted({r["wd"] for r in rows})
    noises = sorted({r["noise"] for r in rows})
    lines = []
    lines.append(f"### Toy (exp_c_grid): {len(rows)} runs, wds={wds}, noises={noises}\n\n")
    lines.append("Mean final test_acc (grok_rate) per (wd, noise):\n\n")
    head = "| wd \\ noise | " + " | ".join(f"{n:g}" for n in noises) + " |\n"
    sep = "|" + "---|" * (len(noises) + 1) + "\n"
    lines.append(head)
    lines.append(sep)
    for wd in wds:
        cells = []
        for noise in noises:
            cell = [r for r in rows
                    if math.isclose(r["wd"], wd) and math.isclose(r["noise"], noise)]
            if not cell:
                cells.append("—")
                continue
            mu = mean([r["final_test_acc"] for r in cell
                       if r["final_test_acc"] is not None])
            grok_rate = sum(1 for r in cell if r["grokked"]) / len(cell)
            cells.append(f"{mu:.3f} ({grok_rate:.0%})")
        lines.append(f"| {wd:g} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    lines.append("Mean final Fourier concentration per (wd, noise):\n\n")
    lines.append(head)
    lines.append(sep)
    for wd in wds:
        cells = []
        for noise in noises:
            cell = [r["final_fourier"] for r in rows
                    if math.isclose(r["wd"], wd) and math.isclose(r["noise"], noise)
                    and r["final_fourier"] is not None]
            cells.append(f"{mean(cell):.3f}" if cell else "—")
        lines.append(f"| {wd:g} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    lines.append("Mean final embedding effective rank per (wd, noise):\n\n")
    lines.append(head)
    lines.append(sep)
    for wd in wds:
        cells = []
        for noise in noises:
            cell = [r["final_rank"] for r in rows
                    if math.isclose(r["wd"], wd) and math.isclose(r["noise"], noise)
                    and r["final_rank"] is not None]
            cells.append(f"{mean(cell):.2f}" if cell else "—")
        lines.append(f"| {wd:g} | " + " | ".join(cells) + " |\n")
    lines.append("\n")
    return lines


def contam_summary_block(rows):
    if not rows:
        return ["No contamination runs found.\n"]
    by_ratio = {}
    for r in rows:
        by_ratio.setdefault(r["ratio_pct"], []).append(r)
    ratios = sorted(by_ratio)
    lines = []
    lines.append(f"### Real (contamination): {len(rows)} runs, ratios={ratios}\n\n")
    lines.append("Mean final metrics per ratio:\n\n")
    metrics = [
        "perplexity",
        "attn_effective_rank",
        "repr_entropy",
        "cos_sim_mean",
        "distinct_2",
        "distinct_3",
        "distinct_4",
    ]
    head = "| ratio_pct | n_seeds | " + " | ".join(metrics) + " |\n"
    sep = "|" + "---|" * (len(metrics) + 2) + "\n"
    lines.append(head)
    lines.append(sep)
    for ratio in ratios:
        cell = by_ratio[ratio]
        n = len(cell)
        vals = []
        for m in metrics:
            xs = [r[m] for r in cell if r.get(m) is not None]
            vals.append(f"{mean(xs):.3f}" if xs else "—")
        lines.append(f"| {ratio} | {n} | " + " | ".join(vals) + " |\n")
    lines.append("\n")
    return lines


def cross_experiment_block(grid_rows, contam_rows):
    """
    The toy and the real experiment share two ideas:
    (1) "rank under stress": embedding effective rank (toy) vs attn_effective_rank (real)
    (2) "structure under stress": Fourier concentration (toy, modular arithmetic specific)
        vs distinct-n / repr_entropy (real, language specific)
    We don't have point-paired data — they are different settings — so we report
    monotonicity-vs-stress separately.
    """
    lines = []
    lines.append("### Cross-experiment: monotonicity of 'rank' under stress\n\n")
    if grid_rows:
        all_x_toy, all_y_toy = [], []
        for r in grid_rows:
            if r["final_rank"] is not None:
                all_x_toy.append(r["noise"])
                all_y_toy.append(r["final_rank"])
        rho_t, p_t = spearman(all_x_toy, all_y_toy)
        if rho_t is not None:
            lines.append(
                f"- Toy: Spearman(noise, embedding_rank) = "
                f"{rho_t:+.3f} (p={p_t:.3g}), n={len(all_x_toy)}.\n"
            )
        else:
            lines.append("- Toy: not enough data for Spearman.\n")
    if contam_rows:
        all_x_real, all_y_real = [], []
        for r in contam_rows:
            if r["attn_effective_rank"] is not None:
                all_x_real.append(r["ratio_pct"])
                all_y_real.append(r["attn_effective_rank"])
        rho_r, p_r = spearman(all_x_real, all_y_real)
        if rho_r is not None:
            lines.append(
                f"- Real: Spearman(ratio_pct, attn_effective_rank) = "
                f"{rho_r:+.3f} (p={p_r:.3g}), n={len(all_x_real)} "
                f"(only {len(set(all_x_real))} ratios so far).\n"
            )
        else:
            lines.append("- Real: not enough data for Spearman (need >= 3 ratios).\n")
    lines.append("\n")

    lines.append("### Cross-experiment: weight-decay rescue\n\n")
    if grid_rows:
        for wd in sorted({r["wd"] for r in grid_rows}):
            for noise in sorted({r["noise"] for r in grid_rows}):
                cell = [r for r in grid_rows
                        if math.isclose(r["wd"], wd) and math.isclose(r["noise"], noise)]
                if not cell:
                    continue
                grok_rate = sum(1 for r in cell if r["grokked"]) / len(cell)
                if 0.0 < grok_rate < 1.0:
                    lines.append(
                        f"- wd={wd:g} noise={noise:g}: partial grokking "
                        f"({grok_rate:.0%}) — rescue boundary.\n"
                    )
    contam_wds = {r["weight_decay"] for r in contam_rows
                  if r.get("weight_decay") is not None}
    if len(contam_wds) <= 1:
        lines.append(
            f"- Real: only weight_decay in {contam_wds} — no wd sweep yet, "
            "can't replicate the rescue. **TODO: rerun contamination at higher wd.**\n"
        )
    lines.append("\n")
    return lines


def overview_plot(grid_rows, contam_rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    if grid_rows:
        wds = sorted({r["wd"] for r in grid_rows})
        noises = sorted({r["noise"] for r in grid_rows})
        palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(wds)))
        for wd, color in zip(wds, palette):
            xs, ys, errs = [], [], []
            for noise in noises:
                cell = [r["final_test_acc"] for r in grid_rows
                        if math.isclose(r["wd"], wd) and math.isclose(r["noise"], noise)
                        and r["final_test_acc"] is not None]
                if not cell:
                    continue
                xs.append(noise)
                ys.append(mean(cell))
                errs.append(stdev(cell) if len(cell) > 1 else 0.0)
            ys_a = np.array(ys)
            errs_a = np.array(errs)
            axes[0].plot(xs, ys_a, marker="o", color=color, label=f"wd={wd:g}")
            axes[0].fill_between(xs, ys_a - errs_a, ys_a + errs_a,
                                 color=color, alpha=0.2)
        axes[0].set_xlabel("noise fraction")
        axes[0].set_ylabel("mean final test acc")
        axes[0].set_title("Toy: test acc vs noise (by wd)")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "no grid runs", ha="center", va="center")
    if contam_rows:
        by_ratio = {}
        for r in contam_rows:
            by_ratio.setdefault(r["ratio_pct"], []).append(r)
        ratios = sorted(by_ratio)
        means = []
        stds = []
        for ratio in ratios:
            xs = [r["attn_effective_rank"] for r in by_ratio[ratio]
                  if r["attn_effective_rank"] is not None]
            means.append(mean(xs) if xs else float("nan"))
            stds.append(stdev(xs) if len(xs) > 1 else 0.0)
        axes[1].errorbar(ratios, means, yerr=stds, fmt="o-",
                         capsize=4, color="C3")
        axes[1].set_xlabel("contamination ratio (%)")
        axes[1].set_ylabel("attn effective rank")
        axes[1].set_title("Real: attn effective rank vs contamination")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "no contamination runs", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary(grid_rows, contam_rows, path):
    lines = []
    lines.append("# Comprehensive analysis: toy grokking + real contamination\n\n")
    lines.append(
        "Two experiments, two stress signals, two outcomes. The toy "
        "tests the **mechanism** (grokking) under label noise; the real "
        "experiment tests the **outcome** (LM degradation) under generative "
        "contamination. They are not point-paired but they should *agree* on "
        "the direction of effect.\n\n"
    )
    lines.append("## Toy summary\n\n")
    lines += grid_summary_block(grid_rows)
    lines.append("## Real summary\n\n")
    lines += contam_summary_block(contam_rows)
    lines.append("## Cross-experiment\n\n")
    lines += cross_experiment_block(grid_rows, contam_rows)

    lines.append("## Reading guide\n\n")
    lines.append(
        "- The toy story: grokking dies above ~10% label noise. wd=1 partially "
        "rescues at 15% (higher accuracy, but no full grok). wd=3 kills "
        "grokking entirely. Fourier concentration tracks grokking; rank tracks "
        "the dimensionality the model uses to fit noise vs structure.\n"
    )
    lines.append(
        "- The real story is incomplete. We need ratios beyond 0/10 and seed "
        "coverage. Once that lands, the equivalent **rank cliff** in "
        "attn_effective_rank should appear at the same place that "
        "perplexity blows up.\n"
    )
    lines.append(
        "- The 'wd rescue' phenomenon in toy predicts that increasing weight "
        "decay during real LM training on contaminated data should also push "
        "the perplexity cliff to higher contamination. **Untested.**\n"
    )
    path.write_text("".join(lines))


def write_csv_summary(grid_rows, contam_rows, path):
    cols = ["experiment", "stressor_name", "stressor_value", "seed",
            "outcome_metric", "outcome_value", "extra"]
    out = []
    for r in grid_rows:
        out.append({
            "experiment": "toy",
            "stressor_name": "noise_fraction",
            "stressor_value": r["noise"],
            "seed": r["seed"],
            "outcome_metric": "final_test_acc",
            "outcome_value": r["final_test_acc"],
            "extra": f"wd={r['wd']:g} grokked={r['grokked']} fourier={r['final_fourier']}",
        })
        out.append({
            "experiment": "toy",
            "stressor_name": "noise_fraction",
            "stressor_value": r["noise"],
            "seed": r["seed"],
            "outcome_metric": "final_embedding_rank",
            "outcome_value": r["final_rank"],
            "extra": f"wd={r['wd']:g}",
        })
    for r in contam_rows:
        out.append({
            "experiment": "real",
            "stressor_name": "ratio_pct",
            "stressor_value": r["ratio_pct"],
            "seed": r["seed"],
            "outcome_metric": "perplexity",
            "outcome_value": r["perplexity"],
            "extra": f"wd={r.get('weight_decay')}",
        })
        out.append({
            "experiment": "real",
            "stressor_name": "ratio_pct",
            "stressor_value": r["ratio_pct"],
            "seed": r["seed"],
            "outcome_metric": "attn_effective_rank",
            "outcome_value": r["attn_effective_rank"],
            "extra": f"wd={r.get('weight_decay')}",
        })
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in out:
            w.writerow(row)


def main():
    grid_rows = load_grid_runs()
    contam_rows = load_contam_runs()
    print(f"[comprehensive] grid runs: {len(grid_rows)}")
    print(f"[comprehensive] contamination runs: {len(contam_rows)}")

    write_summary(grid_rows, contam_rows,
                  OUT_DIR / "comprehensive_summary.md")
    write_csv_summary(grid_rows, contam_rows,
                      OUT_DIR / "comprehensive_summary.csv")
    overview_plot(grid_rows, contam_rows,
                  OUT_DIR / "comprehensive_overview.png")
    print(f"[comprehensive] wrote {OUT_DIR/'comprehensive_summary.md'}")
    print(f"[comprehensive] wrote {OUT_DIR/'comprehensive_summary.csv'}")
    print(f"[comprehensive] wrote {OUT_DIR/'comprehensive_overview.png'}")


if __name__ == "__main__":
    main()
