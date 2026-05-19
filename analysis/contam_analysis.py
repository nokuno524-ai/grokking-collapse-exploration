"""
Analyze contamination training results.

Reads every results/contamination/ratio_<R>_seed_<S>.json, builds a per-run
summary table over (ratio, seed) using the *final* logged step, plots each
mechanistic metric vs contamination ratio with mean +- std error bars, and
tests whether contamination has a monotonic effect on each metric (Spearman
correlation across all (ratio, seed) pairs, plus a Mann-Kendall trend test
on the per-ratio means).

Outputs:
  results/contamination/summary_per_run.csv
  results/contamination/summary_by_ratio.csv
  results/contamination/monotonicity_tests.csv
  results/contamination/plots/<metric>_vs_ratio.png
"""

from __future__ import annotations
import matplotlib.pyplot as plt

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


DEFAULT_DIR = Path("/scratch/qzp4ta/grokking-collapse/results/contamination")
FNAME_RE = re.compile(r"^ratio_(\d+)_seed_(\d+)\.json$")

METRICS = [
    "perplexity",
    "train_loss",
    "attn_effective_rank",
    "repr_entropy",
    "cos_sim_mean",
    "cos_sim_std",
    "distinct_2",
    "distinct_3",
    "distinct_4",
]


def load_runs(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(root.glob("ratio_*_seed_*.json")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        ratio = int(m.group(1))
        seed = int(m.group(2))
        data = json.loads(path.read_text())
        history = data.get("history", [])
        if not history:
            continue
        final = history[-1]
        row = {
            "ratio_pct": ratio,
            "seed": seed,
            "final_step": final.get("step"),
            "n_logged": len(history),
            "weight_decay": data.get("weight_decay"),
            "lr": data.get("lr"),
            "max_steps": data.get("max_steps"),
            "source_file": path.name,
        }
        for k in METRICS:
            row[k] = final.get(k)
        rows.append(row)
    return rows


def group_by_ratio(rows: List[Dict]) -> Dict[int, List[Dict]]:
    by_ratio: Dict[int, List[Dict]] = {}
    for r in rows:
        by_ratio.setdefault(r["ratio_pct"], []).append(r)
    return by_ratio


def per_ratio_summary(rows: List[Dict]) -> List[Dict]:
    by_ratio = group_by_ratio(rows)
    out: List[Dict] = []
    for ratio in sorted(by_ratio):
        runs = by_ratio[ratio]
        agg = {"ratio_pct": ratio, "n_seeds": len(runs)}
        for k in METRICS:
            vals = [r[k] for r in runs if r.get(k) is not None]
            if not vals:
                agg[f"{k}_mean"] = None
                agg[f"{k}_std"] = None
                agg[f"{k}_n"] = 0
                continue
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg[f"{k}_n"] = len(vals)
        out.append(agg)
    return out


def spearman_fallback(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spearman rank correlation with rough p-value (used if scipy missing)."""
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0, 1.0
    rho = float(np.corrcoef(rx, ry)[0, 1])
    # Approximate two-sided p via Fisher transform; ok for n >= 10.
    if abs(rho) >= 1.0:
        return rho, 0.0
    z = math.atanh(rho) * math.sqrt((n - 3))
    # Two-sided normal p
    p = math.erfc(abs(z) / math.sqrt(2))
    return rho, p


def mann_kendall(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Two-sided Mann-Kendall trend test on an ordered series.
    Returns (S, tau, p). Suitable for short ordered series (per-ratio means).
    """
    n = len(values)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = values[j] - values[i]
            if d > 0:
                s += 1
            elif d < 0:
                s -= 1
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    tau = s / (0.5 * n * (n - 1)) if n > 1 else float("nan")
    return float(s), float(tau), float(p)


def monotonicity_tests(rows: List[Dict], summary: List[Dict]) -> List[Dict]:
    """For each metric: Spearman across all (ratio, seed) and MK on per-ratio means."""
    out: List[Dict] = []
    by_ratio = group_by_ratio(rows)
    sorted_ratios = sorted(by_ratio)

    for k in METRICS:
        all_x: List[float] = []
        all_y: List[float] = []
        for ratio in sorted_ratios:
            for r in by_ratio[ratio]:
                v = r.get(k)
                if v is None:
                    continue
                all_x.append(float(ratio))
                all_y.append(float(v))
        if len(all_x) < 3:
            out.append({
                "metric": k,
                "n_points": len(all_x),
                "spearman_rho": None,
                "spearman_p": None,
                "mk_S": None,
                "mk_tau": None,
                "mk_p": None,
                "n_ratios": len(sorted_ratios),
            })
            continue
        x_arr = np.array(all_x)
        y_arr = np.array(all_y)
        if scipy_stats is not None:
            rho_res = scipy_stats.spearmanr(x_arr, y_arr)
            rho = float(rho_res.correlation) if hasattr(rho_res, "correlation") \
                else float(rho_res.statistic)
            rho_p = float(rho_res.pvalue)
        else:
            rho, rho_p = spearman_fallback(x_arr, y_arr)
        means = []
        for ratio in sorted_ratios:
            row = next((s for s in summary if s["ratio_pct"] == ratio), None)
            mv = row.get(f"{k}_mean") if row else None
            if mv is not None:
                means.append(mv)
        if len(means) >= 3:
            S, tau, mk_p = mann_kendall(np.array(means))
        else:
            S, tau, mk_p = float("nan"), float("nan"), float("nan")
        out.append({
            "metric": k,
            "n_points": len(all_x),
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "mk_S": S,
            "mk_tau": tau,
            "mk_p": mk_p,
            "n_ratios": len(sorted_ratios),
        })
    return out


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def plot_metric(
    metric: str,
    summary: List[Dict],
    rows: List[Dict],
    out_path: Path,
    test_row: Dict,
) -> None:
    by_ratio = group_by_ratio(rows)
    ratios = sorted(by_ratio)
    means: List[float] = []
    stds: List[float] = []
    for r in ratios:
        s = next((x for x in summary if x["ratio_pct"] == r), None)
        if s is None:
            means.append(float("nan"))
            stds.append(0.0)
            continue
        means.append(s.get(f"{metric}_mean") if s.get(f"{metric}_mean") is not None else float("nan"))
        stds.append(s.get(f"{metric}_std") if s.get(f"{metric}_std") is not None else 0.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        ratios, means, yerr=stds, fmt="o-", capsize=4, lw=1.6,
        markersize=6, label="mean +- std",
    )
    for r in ratios:
        for run in by_ratio[r]:
            v = run.get(metric)
            if v is None:
                continue
            ax.scatter(r, v, color="gray", alpha=0.45, s=18, zorder=2)
    ax.set_xlabel("Contamination ratio (%)")
    ax.set_ylabel(metric)
    title = f"{metric} vs contamination ratio"
    rho = test_row.get("spearman_rho")
    rho_p = test_row.get("spearman_p")
    if rho is not None and not (isinstance(rho, float) and math.isnan(rho)):
        title += f"\nSpearman rho={rho:.3f}, p={rho_p:.3g}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--plot-dir", type=Path, default=None)
    args = ap.parse_args()

    root: Path = args.root
    plot_dir: Path = args.plot_dir or (root / "plots")

    rows = load_runs(root)
    print(f"Loaded {len(rows)} run(s) from {root}")
    if not rows:
        return

    summary = per_ratio_summary(rows)
    tests = monotonicity_tests(rows, summary)

    per_run_fields = [
        "ratio_pct", "seed", "final_step", "n_logged",
        "weight_decay", "lr", "max_steps", "source_file",
    ] + METRICS
    by_ratio_fields = ["ratio_pct", "n_seeds"]
    for k in METRICS:
        by_ratio_fields += [f"{k}_mean", f"{k}_std", f"{k}_n"]
    test_fields = ["metric", "n_points", "n_ratios",
                   "spearman_rho", "spearman_p",
                   "mk_S", "mk_tau", "mk_p"]

    write_csv(root / "summary_per_run.csv", rows, per_run_fields)
    write_csv(root / "summary_by_ratio.csv", summary, by_ratio_fields)
    write_csv(root / "monotonicity_tests.csv", tests, test_fields)
    print(f"Wrote {root/'summary_per_run.csv'}")
    print(f"Wrote {root/'summary_by_ratio.csv'}")
    print(f"Wrote {root/'monotonicity_tests.csv'}")

    test_by_metric = {t["metric"]: t for t in tests}
    for k in METRICS:
        out_path = plot_dir / f"{k}_vs_ratio.png"
        plot_metric(k, summary, rows, out_path, test_by_metric.get(k, {}))
        print(f"Wrote {out_path}")

    print("\n# Monotonicity tests (Spearman across runs, Mann-Kendall on per-ratio means)")
    print("metric                 | n_pts | rho     p_rho     | tau     p_mk")
    print("-" * 70)
    for t in tests:
        rho = t.get("spearman_rho")
        rho_p = t.get("spearman_p")
        tau = t.get("mk_tau")
        mk_p = t.get("mk_p")

        def _fmt(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "  -   "
            return f"{v: .3f}"
        print(f"{t['metric']:22s} | {t['n_points']:5d} | "
              f"{_fmt(rho)} {_fmt(rho_p)}  | {_fmt(tau)} {_fmt(mk_p)}")


if __name__ == "__main__":
    main()
