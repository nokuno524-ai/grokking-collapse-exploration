"""
Deep analysis of the wd x noise grid (Experiment C, 90 runs).

Loads results.json under
  results/exp_c_grid/wd<W>/noise<N>/seed_<S>/results.json
and produces:

  analysis/exp_c_grid_summary.csv         (per-run rows)
  analysis/exp_c_grid_by_cell.csv         (mean/std per (wd, noise))
  analysis/exp_c_grid_summary.md          (human-readable summary)
  analysis/exp_c_grid_heatmap_<metric>.png
  analysis/exp_c_grid_fourier_curves.png  (fourier vs noise, faceted by wd)
  analysis/exp_c_grid_rank_curves.png     (effective rank vs noise, faceted by wd)
  analysis/exp_c_grid_trajectories_wd1_n015.png
                                          (per-seed test_acc / train_acc trajectories
                                           for the wd=1 x noise=0.15 cell)
  analysis/exp_c_grid_trajectory_signatures.csv
                                          (final-step rank, fourier, weight_norm,
                                           late-stage slope of test_acc, max test_acc,
                                           and a "memorization vs grokking" tag for
                                           every seed)

The point is to answer three questions:
  1. Does wd shift the Fourier cliff to higher noise?
  2. For wd=1 x noise=0.15, are the high-acc seeds slowly grokking or stuck
     in memorization? (slope of test_acc over the last quartile of training)
  3. Does wd preserve embedding effective rank under noise?
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

import matplotlib

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_ROOT = PROJECT_ROOT / "results" / "exp_c_grid"
OUT_DIR = PROJECT_ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WD_RE = re.compile(r"^wd(?P<wd>[\d.]+)$")
NOISE_RE = re.compile(r"^noise(?P<noise>[\d.]+)$")
SEED_RE = re.compile(r"^seed_(?P<seed>\d+)$")


def safe_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def collect_runs():
    rows = []
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
                results_path = seed_dir / "results.json"
                if not results_path.exists():
                    continue
                with results_path.open() as f:
                    data = json.load(f)
                history = data.get("history", []) or []
                rows.append({
                    "wd": wd,
                    "noise": noise,
                    "seed": seed,
                    "grokked": bool(data.get("grokked", False)),
                    "grokking_step": data.get("grokking_step"),
                    "final_train_acc": safe_float(data.get("final_train_acc")),
                    "final_test_acc": safe_float(data.get("final_test_acc")),
                    "final_weight_norm": safe_float(data.get("final_weight_norm")),
                    "final_embedding_rank": safe_float(data.get("final_embedding_rank")),
                    "final_fourier_concentration": safe_float(
                        data.get("final_fourier_concentration")
                    ),
                    "history": history,
                    "results_path": str(results_path),
                })
    return rows


def trajectory_signature(history):
    """
    Late-stage diagnostic: compares avg test_acc in the final quartile of training
    against the second-to-last quartile. Positive slope (>0.02) and not yet grokked
    suggests slow grokking; flat (|slope|<0.005) at high acc suggests stuck-memorization.
    """
    if not history:
        return None
    [h["step"] for h in history]
    accs = [h.get("test_acc", 0.0) for h in history]
    train_accs = [h.get("train_acc", 0.0) for h in history]
    n = len(history)
    if n < 8:
        return {
            "max_test_acc": max(accs) if accs else None,
            "final_test_acc": accs[-1] if accs else None,
            "final_train_acc": train_accs[-1] if train_accs else None,
            "late_slope": None,
            "tag": "too_short",
        }
    q3 = mean(accs[3 * n // 4:])
    q2 = mean(accs[n // 2: 3 * n // 4])
    slope = q3 - q2
    final_test = accs[-1]
    final_train = train_accs[-1]
    max_test = max(accs)
    if final_test >= 0.95:
        tag = "grokked"
    elif slope > 0.02 and final_test < 0.95:
        tag = "slow_grokking"
    elif final_train > 0.95 and final_test < 0.5:
        tag = "memorization"
    elif final_train > 0.95 and final_test < 0.95:
        tag = "high_acc_memorization"
    elif final_test < 0.2:
        tag = "stuck_low"
    else:
        tag = "intermediate"
    return {
        "max_test_acc": max_test,
        "final_test_acc": final_test,
        "final_train_acc": final_train,
        "late_slope": slope,
        "tag": tag,
    }


def aggregate(rows, wds, noises):
    agg = {}
    for wd in wds:
        for noise in noises:
            cell = [
                r for r in rows
                if math.isclose(r["wd"], wd) and math.isclose(r["noise"], noise)
            ]
            test_accs = [r["final_test_acc"] for r in cell if r["final_test_acc"] is not None]
            train_accs = [r["final_train_acc"] for r in cell if r["final_train_acc"] is not None]
            fouriers = [
                r["final_fourier_concentration"] for r in cell
                if r["final_fourier_concentration"] is not None
            ]
            ranks = [
                r["final_embedding_rank"] for r in cell
                if r["final_embedding_rank"] is not None
            ]
            wnorms = [
                r["final_weight_norm"] for r in cell
                if r["final_weight_norm"] is not None
            ]
            steps = [r["grokking_step"] for r in cell if r.get("grokking_step") is not None]
            agg[(wd, noise)] = {
                "n": len(cell),
                "grok_count": sum(1 for r in cell if r["grokked"]),
                "grok_rate": (sum(1 for r in cell if r["grokked"]) / len(cell)) if cell else 0.0,
                "test_acc_mean": mean(test_accs) if test_accs else float("nan"),
                "test_acc_std": stdev(test_accs) if len(test_accs) > 1 else 0.0,
                "train_acc_mean": mean(train_accs) if train_accs else float("nan"),
                "fourier_mean": mean(fouriers) if fouriers else float("nan"),
                "fourier_std": stdev(fouriers) if len(fouriers) > 1 else 0.0,
                "rank_mean": mean(ranks) if ranks else float("nan"),
                "rank_std": stdev(ranks) if len(ranks) > 1 else 0.0,
                "weight_norm_mean": mean(wnorms) if wnorms else float("nan"),
                "grok_step_median": float(np.median(steps)) if steps else None,
            }
    return agg


def heatmap(matrix, wds, noises, title, cbar_label, out_path,
            cmap="viridis", fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, origin="upper")
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([f"{n:g}" for n in noises])
    ax.set_yticks(range(len(wds)))
    ax.set_yticklabels([f"{w:g}" for w in wds])
    ax.set_xlabel("noise fraction")
    ax.set_ylabel("weight decay")
    ax.set_title(title)
    finite = matrix[np.isfinite(matrix)]
    mid = (finite.max() + finite.min()) / 2 if finite.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v),
                        ha="center", va="center",
                        color="white" if v < mid else "black",
                        fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def line_plot_by_wd(rows, wds, noises, metric, out_path, ylabel, title):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(wds)))
    for wd, color in zip(wds, palette):
        means, stds, xs = [], [], []
        for noise in noises:
            cell = [
                r[metric] for r in rows
                if math.isclose(r["wd"], wd) and math.isclose(r["noise"], noise)
                and r.get(metric) is not None
            ]
            if not cell:
                continue
            xs.append(noise)
            means.append(mean(cell))
            stds.append(stdev(cell) if len(cell) > 1 else 0.0)
        means_a = np.array(means)
        stds_a = np.array(stds)
        ax.plot(xs, means_a, marker="o", color=color, label=f"wd={wd:g}")
        ax.fill_between(xs, means_a - stds_a, means_a + stds_a, color=color, alpha=0.18)
    ax.set_xlabel("noise fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def find_fourier_cliff(rows, wds, noises, threshold=0.20):
    """
    For each wd, return the smallest noise where mean fourier drops below threshold.
    Threshold ~0.20 reflects the empirical clean-grokked range (~0.27).
    """
    cliff = {}
    for wd in wds:
        for noise in noises:
            cell = [
                r["final_fourier_concentration"] for r in rows
                if math.isclose(r["wd"], wd) and math.isclose(r["noise"], noise)
                and r["final_fourier_concentration"] is not None
            ]
            if not cell:
                continue
            mu = mean(cell)
            if mu < threshold:
                cliff[wd] = noise
                break
        cliff.setdefault(wd, None)
    return cliff


def plot_wd1_n015_trajectories(rows, out_path):
    cell = [
        r for r in rows
        if math.isclose(r["wd"], 1.0) and math.isclose(r["noise"], 0.15)
    ]
    if not cell:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(cell))))
    for r, color in zip(sorted(cell, key=lambda x: x["seed"]), palette):
        history = r["history"]
        steps = [h["step"] for h in history]
        train = [h.get("train_acc", 0.0) for h in history]
        test = [h.get("test_acc", 0.0) for h in history]
        axes[0].plot(steps, train, color=color, alpha=0.85,
                     label=f"seed={r['seed']}")
        axes[1].plot(steps, test, color=color, alpha=0.85,
                     label=f"seed={r['seed']}")
    for ax in axes:
        ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5,
                   label="grokking threshold")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("train accuracy")
    axes[0].set_title("wd=1 x noise=0.15: train accuracy")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("wd=1 x noise=0.15: test accuracy")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_test_acc_late_slope(rows, wds, noises, out_path):
    """Final-quartile minus second-to-last-quartile mean test_acc, per cell."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(wds)))
    for wd, color in zip(wds, palette):
        means = []
        xs = []
        for noise in noises:
            cell = [
                trajectory_signature(r["history"]) for r in rows
                if math.isclose(r["wd"], wd) and math.isclose(r["noise"], noise)
            ]
            slopes = [s["late_slope"] for s in cell
                      if s and s.get("late_slope") is not None]
            if not slopes:
                continue
            xs.append(noise)
            means.append(mean(slopes))
        ax.plot(xs, means, marker="o", color=color, label=f"wd={wd:g}")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("noise fraction")
    ax.set_ylabel("late-stage test_acc slope (Q4 - Q3 mean)")
    ax.set_title("Late-stage trajectory slope: positive = still climbing")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary_md(rows, agg, wds, noises, cliff, signatures, path):
    lines = []
    lines.append("# Experiment C grid: wd x noise (90 runs)\n\n")
    lines.append(f"Loaded {len(rows)} runs from `{GRID_ROOT}`.\n\n")

    def cell_str(wd, noise, key, fmt="{:.3f}"):
        a = agg[(wd, noise)]
        v = a[key]
        if isinstance(v, float) and math.isnan(v):
            return "—"
        return fmt.format(v)

    lines.append("## Mean final test accuracy (rows=wd, cols=noise)\n\n")
    head = "| wd \\ noise | " + " | ".join(f"{n:g}" for n in noises) + " |\n"
    sep = "|" + "---|" * (len(noises) + 1) + "\n"
    lines.append(head)
    lines.append(sep)
    for wd in wds:
        cells = [cell_str(wd, n, "test_acc_mean") for n in noises]
        lines.append(f"| {wd:g} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    lines.append("## Grok rate (fraction of seeds that crossed 0.95 test acc)\n\n")
    lines.append(head)
    lines.append(sep)
    for wd in wds:
        cells = [cell_str(wd, n, "grok_rate", fmt="{:.2f}") for n in noises]
        lines.append(f"| {wd:g} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    lines.append("## Mean final Fourier concentration\n\n")
    lines.append(head)
    lines.append(sep)
    for wd in wds:
        cells = [cell_str(wd, n, "fourier_mean") for n in noises]
        lines.append(f"| {wd:g} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    lines.append("## Mean final embedding effective rank\n\n")
    lines.append(head)
    lines.append(sep)
    for wd in wds:
        cells = [cell_str(wd, n, "rank_mean", fmt="{:.2f}") for n in noises]
        lines.append(f"| {wd:g} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    lines.append("## Mean final weight norm\n\n")
    lines.append(head)
    lines.append(sep)
    for wd in wds:
        cells = [cell_str(wd, n, "weight_norm_mean", fmt="{:.2f}") for n in noises]
        lines.append(f"| {wd:g} | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    lines.append("## Fourier cliff (smallest noise where mean fourier < 0.20)\n\n")
    for wd, n in cliff.items():
        n_str = "no crossing" if n is None else f"{n:g}"
        lines.append(f"- wd={wd:g}: {n_str}\n")
    lines.append("\n")

    lines.append("## wd=1 x noise=0.15 trajectory tags (does it slowly grok?)\n\n")
    lines.append("| seed | tag | final_train | final_test | max_test | late_slope (Q4-Q3) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for s in signatures:
        if not (math.isclose(s["wd"], 1.0) and math.isclose(s["noise"], 0.15)):
            continue
        lines.append(
            f"| {s['seed']} | {s['tag']} | {s['final_train_acc']:.3f} | "
            f"{s['final_test_acc']:.3f} | {s['max_test_acc']:.3f} | "
            f"{(s['late_slope'] if s['late_slope'] is not None else 0):+.4f} |\n"
        )
    lines.append("\n")

    lines.append("## Per-seed table (sorted by wd, noise, seed)\n\n")
    lines.append("| wd | noise | seed | grokked | step | test | train | fourier | rank | wnorm |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(rows, key=lambda x: (x["wd"], x["noise"], x["seed"])):
        step = r["grokking_step"]
        step_s = "—" if step is None else str(int(step))
        lines.append(
            f"| {r['wd']:g} | {r['noise']:g} | {r['seed']} | {r['grokked']} | {step_s} | "
            f"{r['final_test_acc']:.3f} | {r['final_train_acc']:.3f} | "
            f"{r['final_fourier_concentration']:.3f} | "
            f"{r['final_embedding_rank']:.2f} | {r['final_weight_norm']:.2f} |\n"
        )

    path.write_text("".join(lines))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    rows = collect_runs()
    print(f"[exp_c_grid] loaded {len(rows)} runs from {GRID_ROOT}")
    if not rows:
        raise SystemExit("no runs found")

    wds = sorted({r["wd"] for r in rows})
    noises = sorted({r["noise"] for r in rows})
    print(f"[exp_c_grid] wds={wds}  noises={noises}")

    agg = aggregate(rows, wds, noises)

    # Per-run CSV
    per_run_fields = [
        "wd", "noise", "seed",
        "grokked", "grokking_step",
        "final_train_acc", "final_test_acc",
        "final_fourier_concentration",
        "final_embedding_rank",
        "final_weight_norm",
    ]
    write_csv(OUT_DIR / "exp_c_grid_summary.csv", rows, per_run_fields)
    print(f"[exp_c_grid] wrote {OUT_DIR/'exp_c_grid_summary.csv'}")

    # Per-cell CSV
    cell_rows = []
    for (wd, noise), a in agg.items():
        cell_rows.append({"wd": wd, "noise": noise, **a})
    cell_fields = [
        "wd", "noise", "n", "grok_count", "grok_rate",
        "test_acc_mean", "test_acc_std",
        "train_acc_mean",
        "fourier_mean", "fourier_std",
        "rank_mean", "rank_std",
        "weight_norm_mean",
        "grok_step_median",
    ]
    write_csv(OUT_DIR / "exp_c_grid_by_cell.csv", cell_rows, cell_fields)
    print(f"[exp_c_grid] wrote {OUT_DIR/'exp_c_grid_by_cell.csv'}")

    # Trajectory signatures
    signatures = []
    for r in rows:
        sig = trajectory_signature(r["history"]) or {}
        signatures.append({
            "wd": r["wd"], "noise": r["noise"], "seed": r["seed"],
            **sig,
        })
    sig_fields = ["wd", "noise", "seed",
                  "final_train_acc", "final_test_acc", "max_test_acc",
                  "late_slope", "tag"]
    write_csv(OUT_DIR / "exp_c_grid_trajectory_signatures.csv",
              signatures, sig_fields)
    print(f"[exp_c_grid] wrote {OUT_DIR/'exp_c_grid_trajectory_signatures.csv'}")

    # Fourier cliff
    cliff = find_fourier_cliff(rows, wds, noises, threshold=0.20)
    print(f"[exp_c_grid] fourier cliff (threshold=0.20): {cliff}")

    # Heatmaps
    nW, nN = len(wds), len(noises)
    test_mat = np.full((nW, nN), np.nan)
    fourier_mat = np.full((nW, nN), np.nan)
    rank_mat = np.full((nW, nN), np.nan)
    grok_rate_mat = np.full((nW, nN), np.nan)
    for i, wd in enumerate(wds):
        for j, noise in enumerate(noises):
            a = agg[(wd, noise)]
            test_mat[i, j] = a["test_acc_mean"]
            fourier_mat[i, j] = a["fourier_mean"]
            rank_mat[i, j] = a["rank_mean"]
            grok_rate_mat[i, j] = a["grok_rate"]
    heatmap(test_mat, wds, noises,
            "Mean final test accuracy", "test acc",
            OUT_DIR / "exp_c_grid_heatmap_test_acc.png", fmt="{:.3f}")
    heatmap(fourier_mat, wds, noises,
            "Mean final Fourier concentration", "fourier",
            OUT_DIR / "exp_c_grid_heatmap_fourier.png", fmt="{:.3f}")
    heatmap(rank_mat, wds, noises,
            "Mean final embedding effective rank", "rank",
            OUT_DIR / "exp_c_grid_heatmap_rank.png", fmt="{:.1f}")
    heatmap(grok_rate_mat, wds, noises,
            "Grok rate (fraction with test acc >= 0.95)", "grok rate",
            OUT_DIR / "exp_c_grid_heatmap_grok_rate.png", fmt="{:.2f}")

    # Line plots: fourier and rank vs noise, faceted by wd
    line_plot_by_wd(rows, wds, noises,
                    "final_fourier_concentration",
                    OUT_DIR / "exp_c_grid_fourier_curves.png",
                    ylabel="final Fourier concentration",
                    title="Fourier concentration vs noise (by wd)")
    line_plot_by_wd(rows, wds, noises,
                    "final_embedding_rank",
                    OUT_DIR / "exp_c_grid_rank_curves.png",
                    ylabel="final embedding effective rank",
                    title="Effective embedding rank vs noise (by wd)")

    # Trajectories for the wd=1 x noise=0.15 cell
    plot_wd1_n015_trajectories(
        rows, OUT_DIR / "exp_c_grid_trajectories_wd1_n015.png"
    )
    plot_test_acc_late_slope(
        rows, wds, noises, OUT_DIR / "exp_c_grid_late_slope.png"
    )

    # Markdown summary
    write_summary_md(rows, agg, wds, noises, cliff, signatures,
                     OUT_DIR / "exp_c_grid_summary.md")
    print(f"[exp_c_grid] wrote {OUT_DIR/'exp_c_grid_summary.md'}")
    print(f"[exp_c_grid] all artifacts saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
