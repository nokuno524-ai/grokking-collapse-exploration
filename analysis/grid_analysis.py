"""
Analyze the level x severity x seed grid sweep.

Reads every results.json under results/grid/level<L>_sev<S>/seed_<seed>/,
writes a CSV summary, and produces three plots:
  - phase-transition heatmap (level x severity, mean test_acc)
  - fourier concentration vs severity, faceted by level
  - grokking-step heatmap (level x severity, median grokking_step)

Run with the project venv:  bash run_in_venv.sh python3 analysis/grid_analysis.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_ROOT = PROJECT_ROOT / "results" / "grid"
OUT_DIR = PROJECT_ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIR_RE = re.compile(r"^level(?P<level>[\d.]+)_sev(?P<sev>[\d.]+)$")
SEED_RE = re.compile(r"^seed_(?P<seed>\d+)$")


def _safe_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def collect_runs():
    rows = []
    for cond_dir in sorted(GRID_ROOT.iterdir()):
        m = DIR_RE.match(cond_dir.name)
        if not m or not cond_dir.is_dir():
            continue
        level = float(m.group("level"))
        sev = float(m.group("sev"))
        for seed_dir in sorted(cond_dir.iterdir()):
            sm = SEED_RE.match(seed_dir.name)
            if not sm or not seed_dir.is_dir():
                continue
            seed = int(sm.group("seed"))
            results_path = seed_dir / "results.json"
            if not results_path.exists():
                continue
            with results_path.open() as f:
                data = json.load(f)
            rows.append(
                {
                    "level": level,
                    "severity": sev,
                    "seed": seed,
                    "grokked": bool(data.get("grokked", False)),
                    "grokking_step": data.get("grokking_step"),
                    "final_train_acc": _safe_float(data.get("final_train_acc")),
                    "final_test_acc": _safe_float(data.get("final_test_acc")),
                    "final_fourier_concentration": _safe_float(
                        data.get("final_fourier_concentration")
                    ),
                    "final_weight_norm": _safe_float(data.get("final_weight_norm")),
                    "final_embedding_rank": _safe_float(
                        data.get("final_embedding_rank")
                    ),
                }
            )
    return rows


from grokkit.parser import collect_runs as _cr
def collect_runs():
    rows = _cr(GRID_ROOT)
    # alias keys for legacy scripts
    for r in rows:
        if "collapse_level" in r: r["level"] = r["collapse_level"]
        if "collapse_severity" in r: r["severity"] = r["collapse_severity"]
    return rows

def write_csv(rows, path):
    cols = [
        "level",
        "severity",
        "seed",
        "grokked",
        "grokking_step",
        "final_train_acc",
        "final_test_acc",
        "final_fourier_concentration",
        "final_weight_norm",
        "final_embedding_rank",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate(rows, levels, severities):
    """For each (level, severity) cell, compute mean/std across seeds."""
    agg = {}
    for L in levels:
        for S in severities:
            cell = [
                r
                for r in rows
                if math.isclose(r["level"], L) and math.isclose(r["severity"], S)
            ]
            test_accs = [
                r["final_test_acc"] for r in cell if r["final_test_acc"] is not None
            ]
            fouriers = [
                r["final_fourier_concentration"]
                for r in cell
                if r["final_fourier_concentration"] is not None
            ]
            steps = [
                r["grokking_step"]
                for r in cell
                if r.get("grokking_step") is not None
            ]
            agg[(L, S)] = {
                "n": len(cell),
                "test_acc_mean": mean(test_accs) if test_accs else float("nan"),
                "test_acc_std": stdev(test_accs) if len(test_accs) > 1 else 0.0,
                "fourier_mean": mean(fouriers) if fouriers else float("nan"),
                "fourier_std": stdev(fouriers) if len(fouriers) > 1 else 0.0,
                "grok_step_median": median(steps) if steps else None,
                "grok_rate": (
                    sum(1 for r in cell if r["grokked"]) / len(cell) if cell else 0.0
                ),
            }
    return agg


def write_summary_md(rows, agg, levels, severities, path):
    lines = []
    lines.append("# Grid Sweep Summary (level × severity × seed)\n")
    lines.append(f"Total runs collected: **{len(rows)}**\n")
    cells = {(L, S): [r for r in rows if math.isclose(r['level'], L) and math.isclose(r['severity'], S)] for L in levels for S in severities}
    missing = [k for k, v in cells.items() if len(v) < 5]
    if missing:
        lines.append("Cells with <5 seeds:\n")
        for L, S in missing:
            n = len(cells[(L, S)])
            lines.append(f"- level={L} severity={S}: {n}/5\n")
    else:
        lines.append("All cells have 5 seeds.\n")
    lines.append("\n## Mean test_acc (rows=level, cols=severity)\n\n")
    header = "| level \\ sev | " + " | ".join(f"{S}" for S in severities) + " |"
    sep = "|" + "---|" * (len(severities) + 1)
    lines.append(header + "\n")
    lines.append(sep + "\n")
    for L in levels:
        cells_str = []
        for S in severities:
            a = agg[(L, S)]
            cells_str.append(f"{a['test_acc_mean']:.3f}±{a['test_acc_std']:.3f}")
        lines.append(f"| {L} | " + " | ".join(cells_str) + " |\n")

    lines.append("\n## Mean fourier concentration\n\n")
    lines.append(header + "\n")
    lines.append(sep + "\n")
    for L in levels:
        cells_str = []
        for S in severities:
            a = agg[(L, S)]
            cells_str.append(f"{a['fourier_mean']:.3f}±{a['fourier_std']:.3f}")
        lines.append(f"| {L} | " + " | ".join(cells_str) + " |\n")

    lines.append("\n## Median grokking step (— = none grokked in cell)\n\n")
    lines.append(header + "\n")
    lines.append(sep + "\n")
    for L in levels:
        cells_str = []
        for S in severities:
            step = agg[(L, S)]["grok_step_median"]
            cells_str.append("—" if step is None else str(int(step)))
        lines.append(f"| {L} | " + " | ".join(cells_str) + " |\n")

    lines.append("\n## Grok rate (fraction of seeds that grokked)\n\n")
    lines.append(header + "\n")
    lines.append(sep + "\n")
    for L in levels:
        cells_str = []
        for S in severities:
            cells_str.append(f"{agg[(L, S)]['grok_rate']:.2f}")
        lines.append(f"| {L} | " + " | ".join(cells_str) + " |\n")

    lines.append("\n## Per-seed table\n\n")
    lines.append("| level | severity | seed | grokked | grokking_step | test_acc | fourier |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in sorted(rows, key=lambda x: (x["level"], x["severity"], x["seed"])):
        step = r["grokking_step"]
        step_s = "—" if step is None else str(int(step))
        lines.append(
            f"| {r['level']} | {r['severity']} | {r['seed']} | "
            f"{r['grokked']} | {step_s} | "
            f"{r['final_test_acc']:.3f} | {r['final_fourier_concentration']:.3f} |\n"
        )

    path.write_text("".join(lines))


def heatmap(matrix, levels, severities, title, cbar_label, out_path,
            cmap="viridis", fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, origin="upper")
    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([f"{S}" for S in severities])
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([f"{L}" for L in levels])
    ax.set_xlabel("severity")
    ax.set_ylabel("contamination level")
    ax.set_title(title)
    for i, _ in enumerate(levels):
        for j, _ in enumerate(severities):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v),
                        ha="center", va="center",
                        color="white" if v < (np.nanmax(matrix) + np.nanmin(matrix)) / 2 else "black",
                        fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_fourier_vs_severity(rows, levels, severities, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(levels)))
    for L, color in zip(levels, palette):
        means, stds, xs = [], [], []
        for S in severities:
            cell = [
                r["final_fourier_concentration"]
                for r in rows
                if math.isclose(r["level"], L)
                and math.isclose(r["severity"], S)
                and r["final_fourier_concentration"] is not None
            ]
            if not cell:
                continue
            xs.append(S)
            means.append(mean(cell))
            stds.append(stdev(cell) if len(cell) > 1 else 0.0)
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(xs, means, marker="o", color=color, label=f"level={L}")
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.2)
    ax.set_xlabel("severity")
    ax.set_ylabel("final Fourier concentration")
    ax.set_title("Fourier concentration vs severity (mean ± std over seeds)")
    ax.legend(title="contamination", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_test_acc_vs_severity(rows, levels, severities, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(levels)))
    for L, color in zip(levels, palette):
        means, stds, xs = [], [], []
        for S in severities:
            cell = [
                r["final_test_acc"]
                for r in rows
                if math.isclose(r["level"], L)
                and math.isclose(r["severity"], S)
                and r["final_test_acc"] is not None
            ]
            if not cell:
                continue
            xs.append(S)
            means.append(mean(cell))
            stds.append(stdev(cell) if len(cell) > 1 else 0.0)
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(xs, means, marker="o", color=color, label=f"level={L}")
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.2)
    ax.set_xlabel("severity")
    ax.set_ylabel("final test accuracy")
    ax.set_title("Test accuracy vs severity (mean ± std over seeds)")
    ax.legend(title="contamination", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    rows = collect_runs()
    print(f"Collected {len(rows)} runs from {GRID_ROOT}")
    if not rows:
        raise SystemExit("No grid results found")

    levels = sorted({r["level"] for r in rows})
    severities = sorted({r["severity"] for r in rows})
    print(f"Levels: {levels}")
    print(f"Severities: {severities}")

    # CSV
    csv_path = OUT_DIR / "grid_summary.csv"
    write_csv(rows, csv_path)
    print(f"Wrote {csv_path}")

    # Aggregates
    agg = aggregate(rows, levels, severities)

    # Markdown summary
    md_path = OUT_DIR / "grid_summary.md"
    write_summary_md(rows, agg, levels, severities, md_path)
    print(f"Wrote {md_path}")

    # Plots
    nL, nS = len(levels), len(severities)
    test_mat = np.full((nL, nS), np.nan)
    fourier_mat = np.full((nL, nS), np.nan)
    grok_step_mat = np.full((nL, nS), np.nan)
    grok_rate_mat = np.full((nL, nS), np.nan)
    for i, L in enumerate(levels):
        for j, S in enumerate(severities):
            a = agg[(L, S)]
            test_mat[i, j] = a["test_acc_mean"]
            fourier_mat[i, j] = a["fourier_mean"]
            if a["grok_step_median"] is not None:
                grok_step_mat[i, j] = a["grok_step_median"]
            grok_rate_mat[i, j] = a["grok_rate"]

    heatmap(test_mat, levels, severities,
            "Phase-transition heatmap: mean test_acc",
            "test accuracy",
            OUT_DIR / "heatmap_test_acc.png", fmt="{:.2f}")
    heatmap(fourier_mat, levels, severities,
            "Fourier concentration heatmap (mean)",
            "fourier concentration",
            OUT_DIR / "heatmap_fourier.png", fmt="{:.2f}")
    heatmap(grok_step_mat, levels, severities,
            "Median grokking step (NaN = none grokked)",
            "step",
            OUT_DIR / "heatmap_grokking_step.png", cmap="magma_r",
            fmt="{:.0f}")
    heatmap(grok_rate_mat, levels, severities,
            "Fraction of seeds that grokked",
            "grok rate",
            OUT_DIR / "heatmap_grok_rate.png", fmt="{:.2f}")

    plot_fourier_vs_severity(rows, levels, severities,
                             OUT_DIR / "fourier_vs_severity.png")
    plot_test_acc_vs_severity(rows, levels, severities,
                              OUT_DIR / "test_acc_vs_severity.png")

    print(f"All artifacts saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
