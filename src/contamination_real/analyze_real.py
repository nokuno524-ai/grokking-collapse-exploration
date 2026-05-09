"""
Aggregate full-realism contamination results: per-(ratio, seed, mode) metric
trajectories from train_real.py + downstream evaluation summary from
eval_downstream.py. Produces:

- training-curve plots per metric (one line per ratio, mean ± std over seeds)
- final-value plots (metric vs contamination ratio with error bars)
- a phase-transition table (largest delta between adjacent ratios)
- comparison plots overlaying the toy experiment results, when available

Run as
------
python -m src.contamination_real.analyze_real \
    --results-dir /scratch/qzp4ta/grokking-collapse/results/contamination_real \
    --toy-results-dir /scratch/qzp4ta/grokking-collapse/results/contamination
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_DIR = Path("/scratch/qzp4ta/grokking-collapse/results/contamination_real")
DEFAULT_TOY_RESULTS_DIR = Path("/scratch/qzp4ta/grokking-collapse/results/contamination")

PLOT_METRICS = [
    "perplexity",
    "repr_rank_last",
    "repr_rank_mean",
    "attn_entropy_mean",
    "feat_density",
    "lora_B_norm_drift",
    "grad_cos_recent_mean",
    "distinct_2",
    "distinct_3",
    "repetition_rate",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(results_dir: Path) -> List[dict]:
    runs = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name in {"summary.json", "downstream_summary.json", "phase_transitions.json"}:
            continue
        try:
            data = json.loads(f.read_text())
            if "history" in data and "ratio_pct" in data:
                runs.append(data)
        except Exception as e:  # noqa: BLE001
            print(f"[analyze_real] could not parse {f}: {e}", flush=True)
    return runs


def group_by(runs: List[dict], key: str) -> Dict:
    groups: Dict = defaultdict(list)
    for r in runs:
        groups[r.get(key)].append(r)
    return dict(groups)


def group_by_ratio(runs: List[dict], mode: str = "ai") -> Dict[int, List[dict]]:
    groups: Dict[int, List[dict]] = defaultdict(list)
    for r in runs:
        if r.get("mode", "ai") != mode:
            continue
        groups[int(r["ratio_pct"])].append(r)
    return dict(sorted(groups.items()))


def final_value(run: dict, metric: str) -> float:
    for entry in reversed(run["history"]):
        if metric in entry:
            try:
                return float(entry[metric])
            except (TypeError, ValueError):
                continue
    return float("nan")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _aligned_curve(
    runs: List[dict], metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    series, steps_ref = [], None
    for r in runs:
        steps = [e["step"] for e in r["history"] if metric in e]
        vals = [e[metric] for e in r["history"] if metric in e]
        if not steps:
            continue
        if steps_ref is None:
            steps_ref = steps
        m = {s: v for s, v in zip(steps, vals)}
        aligned = [m.get(s, np.nan) for s in steps_ref]
        series.append(aligned)
    if not series or steps_ref is None:
        return np.array([]), np.array([]), np.array([])
    arr = np.array(series, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
    return np.array(steps_ref), mean, std


def plot_training_curves(
    groups: Dict[int, List[dict]], plots_dir: Path, suffix: str = ""
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    ratios_sorted = sorted(groups.keys())
    if not ratios_sorted:
        return
    for metric in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        any_data = False
        for i, ratio in enumerate(ratios_sorted):
            color = cmap(i / max(1, len(ratios_sorted) - 1))
            steps, mean, std = _aligned_curve(groups[ratio], metric)
            if steps.size == 0:
                continue
            any_data = True
            ax.plot(steps, mean, lw=2, label=f"{ratio}%", color=color)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xlabel("Training step")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} during training{suffix and ' (' + suffix + ')'}")
        ax.legend(title="ratio", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_name = f"curves_{metric}" + (f"_{suffix}" if suffix else "") + ".png"
        fig.savefig(plots_dir / out_name, dpi=150)
        plt.close(fig)


def plot_final_vs_ratio(
    groups: Dict[int, List[dict]], plots_dir: Path, suffix: str = ""
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric in PLOT_METRICS:
        ratios, means, stds = [], [], []
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
        ax.set_title(f"Final {metric} vs ratio{suffix and ' (' + suffix + ')'}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        name = f"final_{metric}_vs_ratio" + (f"_{suffix}" if suffix else "") + ".png"
        fig.savefig(plots_dir / name, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase transition detection
# ---------------------------------------------------------------------------

def detect_phase_transition(groups: Dict[int, List[dict]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
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


# ---------------------------------------------------------------------------
# Toy / real overlay (the publication-grade comparison plot)
# ---------------------------------------------------------------------------

def overlay_toy(
    real_groups: Dict[int, List[dict]],
    toy_groups: Optional[Dict[int, List[dict]]],
    plots_dir: Path,
    metric: str = "perplexity",
    toy_metric: Optional[str] = None,
) -> None:
    if not real_groups or not toy_groups:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    toy_m = toy_metric or metric
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, groups, m, color in [
        ("Real (GPT-2 medium)", real_groups, metric, "tab:red"),
        ("Toy (mod arithmetic)", toy_groups, toy_m, "tab:blue"),
    ]:
        ratios, means, stds = [], [], []
        for r, runs in sorted(groups.items()):
            vals = [final_value(rn, m) for rn in runs]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue
            ratios.append(r)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        if not ratios:
            continue
        means_n = np.array(means) / max(np.array(means))
        stds_n = np.array(stds) / max(np.array(means))
        ax.errorbar(ratios, means_n, yerr=stds_n, marker="o", capsize=4,
                    lw=2, label=label, color=color)
    ax.set_xlabel("Contamination ratio (%)")
    ax.set_ylabel(f"normalized {metric}")
    ax.set_title(f"Toy vs real: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"overlay_{metric}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--toy-results-dir", type=str,
                        default=str(DEFAULT_TOY_RESULTS_DIR))
    parser.add_argument("--mode", type=str, default="ai",
                        help="Which mode group to analyze (ai|noise|scarcity|self|external)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"

    runs = load_runs(results_dir)
    if not runs:
        print(f"[analyze_real] no runs found in {results_dir}", flush=True)
        return
    print(f"[analyze_real] loaded {len(runs)} runs from {results_dir}", flush=True)

    main_groups = group_by_ratio(runs, mode=args.mode)
    print(f"[analyze_real] mode={args.mode} ratios={list(main_groups.keys())}",
          flush=True)
    plot_training_curves(main_groups, plots_dir, suffix=args.mode)
    plot_final_vs_ratio(main_groups, plots_dir, suffix=args.mode)
    transitions = detect_phase_transition(main_groups)

    # Baselines: noise, scarcity, self, external — emit their own panels.
    extras: Dict[str, Dict[int, List[dict]]] = {}
    for m in ("noise", "scarcity", "self", "external"):
        if m == args.mode:
            continue
        sub = group_by_ratio(runs, mode=m)
        if sub:
            extras[m] = sub
            plot_training_curves(sub, plots_dir, suffix=m)
            plot_final_vs_ratio(sub, plots_dir, suffix=m)

    # Toy overlay
    toy_dir = Path(args.toy_results_dir)
    toy_groups: Optional[Dict[int, List[dict]]] = None
    if toy_dir.exists():
        toy_runs = load_runs(toy_dir)
        if toy_runs:
            toy_groups = group_by_ratio(toy_runs, mode="ai")
        else:
            # toy results may not have 'mode' field; fallback
            toy_groups = defaultdict(list)
            for r in toy_runs:
                toy_groups[int(r.get("ratio_pct", 0))].append(r)
            toy_groups = dict(sorted(toy_groups.items()))
        if toy_groups:
            for metric in ("perplexity", "repr_rank_last", "distinct_3"):
                overlay_toy(main_groups, toy_groups, plots_dir, metric=metric)

    summary = {
        "n_runs": len(runs),
        "mode": args.mode,
        "ratios": list(main_groups.keys()),
        "phase_transitions": transitions,
        "baselines": {
            m: {
                "ratios": list(g.keys()),
                "phase_transitions": detect_phase_transition(g),
            }
            for m, g in extras.items()
        },
    }

    # Pull downstream eval, if available
    ds_path = results_dir / "downstream_summary.json"
    if ds_path.exists():
        try:
            summary["downstream"] = json.loads(ds_path.read_text())
        except Exception:
            pass

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[analyze_real] wrote plots -> {plots_dir} and summary -> {summary_path}",
          flush=True)


if __name__ == "__main__":
    main()
