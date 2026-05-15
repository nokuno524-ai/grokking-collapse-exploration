"""
Load all per-(ratio, seed) result JSONs and produce the phase-transition plots.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_DIR = Path("/scratch/qzp4ta/grokking-collapse/results/contamination")
PLOT_METRICS = [
    "perplexity",
    "attn_effective_rank",
    "repr_entropy",
    "cos_sim_mean",
    "distinct_2",
    "distinct_3",
    "distinct_4",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_runs(results_dir: Path) -> List[dict]:
    runs = []
    for f in sorted(results_dir.glob("ratio_*_seed_*.json")):
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def group_by_ratio(runs: List[dict]) -> Dict[int, List[dict]]:
    groups: Dict[int, List[dict]] = defaultdict(list)
    for r in runs:
        groups[int(r["ratio_pct"])].append(r)
    return dict(sorted(groups.items()))


def final_value(run: dict, metric: str) -> float:
    for entry in reversed(run["history"]):
        if metric in entry:
            return float(entry[metric])
    return float("nan")


# ---------------------------------------------------------------------------
# Plot 1: ratio vs final value (with error bars)
# ---------------------------------------------------------------------------


def plot_final_vs_ratio(groups: Dict[int, List[dict]], plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric in PLOT_METRICS:
        ratios = []
        means = []
        stds = []
        for ratio, runs in groups.items():
            vals = [final_value(r, metric) for r in runs]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue
            ratios.append(ratio)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        if not ratios:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(ratios, means, yerr=stds, marker="o", capsize=4, lw=2)
        ax.set_xlabel("Contamination ratio (%)")
        ax.set_ylabel(metric)
        ax.set_title(f"Final {metric} vs contamination ratio")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"final_{metric}_vs_ratio.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: training curves per metric, one line per ratio (mean across seeds)
# ---------------------------------------------------------------------------


def _aligned_curve(
    runs: List[dict], metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, mean, std) over seeds for a given metric."""
    series = []
    steps_ref = None
    for r in runs:
        steps = [e["step"] for e in r["history"] if metric in e]
        vals = [e[metric] for e in r["history"] if metric in e]
        if steps_ref is None:
            steps_ref = steps
        # Align by step
        m = {s: v for s, v in zip(steps, vals)}
        aligned = [m.get(s, np.nan) for s in steps_ref]
        series.append(aligned)
    arr = np.array(series, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
    return np.array(steps_ref), mean, std


def plot_training_curves(groups: Dict[int, List[dict]], plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    ratios_sorted = sorted(groups.keys())
    for metric in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, ratio in enumerate(ratios_sorted):
            color = cmap(i / max(1, len(ratios_sorted) - 1))
            steps, mean, std = _aligned_curve(groups[ratio], metric)
            ax.plot(steps, mean, lw=2, label=f"{ratio}%", color=color)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
        ax.set_xlabel("Training step")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} during training")
        ax.legend(title="ratio", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"curves_{metric}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: phase-transition detection (largest discrete derivative)
# ---------------------------------------------------------------------------


def detect_phase_transition(groups: Dict[int, List[dict]]) -> Dict[str, dict]:
    """For each metric, find the ratio with the largest |Δmean| between adjacent ratios."""
    out = {}
    ratios = sorted(groups.keys())
    for metric in PLOT_METRICS:
        means = []
        for ratio in ratios:
            vals = [final_value(r, metric) for r in groups[ratio]]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(float(np.mean(vals)) if vals else float("nan"))
        means_arr = np.array(means)
        if len(means_arr) < 2 or np.all(np.isnan(means_arr)):
            continue
        diffs = np.abs(np.diff(means_arr))
        if np.all(np.isnan(diffs)):
            continue
        i = int(np.nanargmax(diffs))
        out[metric] = {
            "transition_between_ratios": [ratios[i], ratios[i + 1]],
            "delta": float(diffs[i]),
            "means_by_ratio": dict(zip(ratios, [float(m) for m in means_arr])),
        }
    return out


def plot_phase_transition(transitions: Dict[str, dict], plots_dir: Path):
    if not transitions:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics = list(transitions.keys())
    deltas = [transitions[m]["delta"] for m in metrics]
    edges = [
        f"{transitions[m]['transition_between_ratios'][0]}→"
        f"{transitions[m]['transition_between_ratios'][1]}%"
        for m in metrics
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(metrics))
    ax.barh(y, deltas, color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    for i, e in enumerate(edges):
        ax.text(deltas[i], i, f"  {e}", va="center", fontsize=9)
    ax.set_xlabel("Largest |Δmean| between adjacent ratios")
    ax.set_title("Phase transition location per metric")
    fig.tight_layout()
    fig.savefig(plots_dir / "phase_transitions.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"

    runs = load_runs(results_dir)
    if not runs:
        print(f"[analyze] no runs found in {results_dir}")
        return
    print(f"[analyze] loaded {len(runs)} runs from {results_dir}")
    groups = group_by_ratio(runs)
    print(f"[analyze] ratios: {list(groups.keys())}")

    plot_final_vs_ratio(groups, plots_dir)
    plot_training_curves(groups, plots_dir)
    transitions = detect_phase_transition(groups)
    plot_phase_transition(transitions, plots_dir)

    summary = {
        "n_runs": len(runs),
        "ratios": list(groups.keys()),
        "phase_transitions": transitions,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze] wrote plots to {plots_dir} and summary to {
            results_dir /
            'summary.json'}")


if __name__ == "__main__":
    main()
