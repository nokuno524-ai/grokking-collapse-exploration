"""
Plot mechanistic metrics vs contamination ratio with mean +/- std error bars
across seeds, using the GPT-2 small contamination runs in
results/contamination/ratio_*_seed_*.json.

Outputs PNGs into results/contamination/plots/.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

DEFAULT_DIR = Path("/scratch/qzp4ta/grokking-collapse/results/contamination")
FNAME_RE = re.compile(r"^ratio_(\d+)_seed_(\d+)\.json$")

PLOT_METRICS = [
    ("perplexity", "Perplexity (held-out)", "log"),
    ("train_loss", "Train loss", "linear"),
    ("attn_effective_rank", "Last-attn effective rank", "linear"),
    ("repr_entropy", "Hidden-state entropy (nats)", "linear"),
    ("cos_sim_mean", "Mean cosine sim", "linear"),
    ("distinct_3", "Distinct-3 (sample diversity)", "linear"),
]


def load_runs(root: Path) -> Dict[int, List[Dict]]:
    by_ratio: Dict[int, List[Dict]] = {}
    for path in sorted(root.glob("ratio_*_seed_*.json")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        ratio = int(m.group(1))
        data = json.loads(path.read_text())
        history = data.get("history", [])
        if not history:
            continue
        by_ratio.setdefault(ratio, []).append({
            "seed": int(m.group(2)),
            "history": history,
        })
    return by_ratio


def collect_final(by_ratio: Dict[int, List[Dict]], key: str):
    ratios, means, stds = [], [], []
    for ratio in sorted(by_ratio):
        vals = [run["history"][-1].get(key) for run in by_ratio[ratio]]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        ratios.append(ratio)
        means.append(statistics.mean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return np.array(ratios), np.array(means), np.array(stds)


def plot_final_vs_ratio(by_ratio, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, label, scale) in zip(axes.flat, PLOT_METRICS):
        x, m, s = collect_final(by_ratio, key)
        if len(x) == 0:
            ax.set_title(f"{label} (no data)")
            ax.axis("off")
            continue
        ax.errorbar(x, m, yerr=s, marker="o", capsize=3, lw=1.4)
        for run_idx, ratio in enumerate(sorted(by_ratio)):
            for run in by_ratio[ratio]:
                v = run["history"][-1].get(key)
                if v is not None:
                    ax.scatter([ratio], [v], s=14, alpha=0.4, color="C1")
        ax.set_xlabel("Contamination ratio (%)")
        ax.set_ylabel(label)
        ax.set_yscale(scale)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    fig.suptitle("Final-step mechanistic metrics vs contamination ratio (real-LM, GPT-2 small)")
    fig.tight_layout()
    out = out_dir / "metrics_vs_ratio.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_trajectories(by_ratio, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cmap = plt.colormaps.get_cmap("viridis")
    ratios = sorted(by_ratio)
    norm = plt.Normalize(min(ratios), max(ratios) if max(ratios) > 0 else 1)
    for ax, (key, label, scale) in zip(axes.flat, PLOT_METRICS):
        for ratio in ratios:
            runs = by_ratio[ratio]
            color = cmap(norm(ratio))
            for run in runs:
                steps = [h.get("step") for h in run["history"]]
                vals = [h.get(key) for h in run["history"]]
                pairs = [(s, v) for s, v in zip(steps, vals) if s is not None and v is not None]
                if not pairs:
                    continue
                xs, ys = zip(*pairs)
                ax.plot(xs, ys, color=color, alpha=0.6, lw=1.0,
                        label=f"ratio={ratio}%" if run is runs[0] else None)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_yscale(scale)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        if ax is axes.flat[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Mechanistic metric trajectories (real-LM, GPT-2 small)")
    fig.tight_layout()
    out = out_dir / "metrics_trajectories.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    out_dir = args.out_dir or (args.root / "plots")
    by_ratio = load_runs(args.root)
    print(f"Ratios with data: {sorted(by_ratio)}; "
          f"n_runs total = {sum(len(v) for v in by_ratio.values())}")
    if not by_ratio:
        return
    plot_final_vs_ratio(by_ratio, out_dir)
    plot_trajectories(by_ratio, out_dir)


if __name__ == "__main__":
    main()
