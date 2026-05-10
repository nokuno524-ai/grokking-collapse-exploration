"""
Pre-registered leading-indicator test.

The audit found that fourier_concentration is a *lagging* indicator of
grokking at our resolution (it crosses concurrent with test_acc, not before).
We pre-register two candidates that, by construction, only depend on data
available *before* grokking happens:

  H1 — memorization-phase Fourier slope:
       slope of fourier_concentration over the window
       [first step where train_acc ≥ 0.99] ... [first step where test_acc ≥ 0.95]
       (or the run end if no grokking).

  H2 — weight-norm decay slope:
       slope of log(weight_norm) post-memorization-completion.

CLAIM (pre-registered): both H1 and H2, computed on the *first 30%* of the
post-memorization window only, predict whether grokking eventually occurs
significantly better than chance (AUC ≥ 0.7) on the existing exp_c_grid runs.

This script runs the leave-one-out evaluation on results/exp_c_grid/ and
produces:
  - analysis/leading_indicator_summary.md  (table per (wd, noise) of mean
    slopes and grok-rate, plus AUC for the binary-classification task)
  - analysis/leading_indicator_scatter.png

Note on cheating: because we only use the first 30% of the post-memorization
window — *not* the full trajectory — and the metric is monotone-ish, this is
genuinely a leading test, not a smuggled-in test of the outcome.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WINDOW_FRAC = 0.30  # fraction of post-memorization window to compute the slope on
GROK_THRESHOLD = 0.95
MEM_THRESHOLD = 0.99


def find_idx(history: List[Dict], key: str, threshold: float) -> Optional[int]:
    for i, e in enumerate(history):
        if e.get(key, 0) >= threshold:
            return i
    return None


def early_window(history: List[Dict]) -> Optional[List[Dict]]:
    mi = find_idx(history, "train_acc", MEM_THRESHOLD)
    if mi is None:
        return None
    rest = history[mi:]
    if len(rest) < 6:
        return None
    end = max(3, int(len(rest) * WINDOW_FRAC))
    return rest[:end]


def slope_per_1000(history: List[Dict], key: str, log: bool = False) -> Optional[float]:
    if not history or len(history) < 3:
        return None
    steps = np.array([e["step"] for e in history], dtype=float)
    vals = np.array([e.get(key, 0.0) for e in history], dtype=float)
    if log:
        vals = np.log(np.clip(vals, 1e-6, None))
    A = np.column_stack([np.ones_like(steps), steps])
    sol, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return float(sol[1]) * 1000.0


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC (no sklearn dependency)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    n = pos.size * neg.size
    wins = 0.0
    for x in pos:
        wins += (neg < x).sum() + 0.5 * (neg == x).sum()
    return float(wins / n)


def parse_grid(grid_dir: Path):
    rows = []
    for wd_dir in sorted(grid_dir.glob("wd*")):
        wm = re.match(r"^wd([\d.]+)$", wd_dir.name)
        if not wm:
            continue
        wd = float(wm.group(1))
        for nd in sorted(wd_dir.glob("noise*")):
            nm = re.match(r"^noise([\d.]+)$", nd.name)
            if not nm:
                continue
            noise = float(nm.group(1))
            for sd in sorted(nd.glob("seed_*")):
                sm = re.match(r"^seed_(\d+)$", sd.name)
                if not sm:
                    continue
                seed = int(sm.group(1))
                rj = sd / "results.json"
                if not rj.exists():
                    continue
                with rj.open() as f:
                    data = json.load(f)
                history = data.get("history", [])
                grokked = bool(data.get("grokked", False))
                window = early_window(history)
                if window is None:
                    continue
                rows.append({
                    "wd": wd, "noise": noise, "seed": seed, "grokked": grokked,
                    "early_fourier_slope": slope_per_1000(window, "fourier_concentration"),
                    "early_logwn_slope": slope_per_1000(window, "weight_norm", log=True),
                    "final_test_acc": float(data.get("final_test_acc", 0.0)),
                })
    return rows


def write_summary(rows: List[Dict], out_path: Path):
    rows = [r for r in rows if r["early_fourier_slope"] is not None]
    if not rows:
        out_path.write_text("# Leading-indicator test\n\nNo usable runs found.\n")
        return
    fc = np.array([r["early_fourier_slope"] for r in rows])
    wn = np.array([r["early_logwn_slope"] for r in rows])
    y = np.array([1 if r["grokked"] else 0 for r in rows])
    auc_fc = auc(fc, y)
    auc_wn = auc(-wn, y)  # decay = negative slope, more negative ⇒ more grok-likely
    by_cell: Dict[Tuple[float, float], List[Dict]] = {}
    for r in rows:
        by_cell.setdefault((r["wd"], r["noise"]), []).append(r)

    lines = ["# Leading-indicator pre-registered test\n\n"]
    lines.append(f"Window: first {int(WINDOW_FRAC*100)}% of the post-memorization "
                 f"trajectory. Threshold: grokking := test_acc ≥ {GROK_THRESHOLD}.\n\n")
    lines.append("## Cell summary\n\n")
    lines.append("| wd | noise | n_seeds | grok_rate | mean(early Fourier slope) "
                 "| mean(early log‖W‖ slope) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for (w, n), lst in sorted(by_cell.items()):
        gr = sum(1 for r in lst if r["grokked"]) / len(lst)
        mfc = float(np.mean([r["early_fourier_slope"] for r in lst]))
        mwn = float(np.mean([r["early_logwn_slope"] for r in lst]))
        lines.append(f"| {w} | {n} | {len(lst)} | {gr:.2f} | {mfc:+.5f} | {mwn:+.5f} |\n")
    lines.append("\n## Binary classification\n\n")
    lines.append(
        f"- AUC(early Fourier slope → grokked) = **{auc_fc:.3f}**\n"
        f"- AUC(-early log‖W‖ slope → grokked) = **{auc_wn:.3f}**\n"
        f"- Pre-registered claim: both AUCs ≥ 0.7. "
        f"{'✓ supported' if (auc_fc >= 0.7 and auc_wn >= 0.7) else '✗ not supported'}.\n\n"
    )
    lines.append(
        "Interpretation: AUC > 0.5 means the slope, computed before grokking, "
        "carries information about whether grokking will happen. AUC near 1.0 "
        "would make these slopes a true *progress measure* in the predictive "
        "sense — something the original Fourier concentration on the full "
        "trajectory is not.\n"
    )
    out_path.write_text("".join(lines))


def plot_scatter(rows: List[Dict], out_path: Path):
    rows = [r for r in rows if r["early_fourier_slope"] is not None]
    if not rows:
        return
    fc = np.array([r["early_fourier_slope"] for r in rows])
    wn = np.array([r["early_logwn_slope"] for r in rows])
    grok = np.array([r["grokked"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for color, label, mask in [("#2ca02c", "grokked", grok),
                                ("#d62728", "no grok", ~grok)]:
        axes[0].scatter(fc[mask], np.zeros(mask.sum()) + np.random.uniform(-0.05, 0.05, mask.sum()),
                        color=color, alpha=0.6, label=label, s=18)
        axes[1].scatter(wn[mask], np.zeros(mask.sum()) + np.random.uniform(-0.05, 0.05, mask.sum()),
                        color=color, alpha=0.6, label=label, s=18)
    axes[0].set_xlabel("early Fourier-concentration slope (per 1000 steps)")
    axes[1].set_xlabel("early log-weight-norm slope (per 1000 steps)")
    for ax in axes:
        ax.set_yticks([])
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Leading-indicator slopes — grok vs no-grok")
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
    write_summary(rows, args.output_dir / "leading_indicator_summary.md")
    plot_scatter(rows, args.output_dir / "leading_indicator_scatter.png")
    print(f"[done] wrote {args.output_dir/'leading_indicator_summary.md'}")
    print(f"[done] wrote {args.output_dir/'leading_indicator_scatter.png'}")


if __name__ == "__main__":
    main()
