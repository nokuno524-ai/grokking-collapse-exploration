"""
Experiment A — causal circuit rescue.

Idea: at the grokking transition certain weight sub-matrices suddenly increase
in *effective* rank (the model uses more directions to encode structure).
This script identifies *which* circuits change.

For one or more runs from results/exp_c_grid/wd<W>/noise<N>/seed_<S>/, it:
  1. iterates the saved checkpoints (every 5k steps),
  2. computes the singular-value spectrum of every weight matrix (Linear,
     Embedding, attention in-proj, attention out-proj, FFN, output head),
  3. per matrix, reports both the full-rank-normalised effective rank
     ((-sum p log p) where p = sigma / sum sigma, then exp of that) and the
     stable rank (||W||_F^2 / ||W||_2^2),
  4. plots the per-matrix rank trajectory vs step,
  5. flags which matrix shows the *sharpest* transition (largest delta in
     effective rank between consecutive checkpoints) and reports the step
     at which it happens. We compare that to the grokking_step recorded
     in results.json.

Then it pairs runs across conditions:
  - "clean grokking" (wd=0.3, noise=0)    — control
  - "wd-rescued" (wd=1.0, noise=0.15 with high but sub-grokking acc) — same circuit?

If the same circuit (e.g. attn in-proj or token embedding) drives the rank
jump in both conditions, that supports the hypothesis that wd doesn't move
the circuit, just its noise tolerance.

Usage:
  python src/causal_circuit_rescue.py \
      --runs results/exp_c_grid/wd0.3/noise0/seed_42 \
             results/exp_c_grid/wd1/noise0.15/seed_42 \
      --output-dir analysis/causal_circuit
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running as `python src/causal_circuit_rescue.py` or as a module.
try:
    from .model import ModularArithmeticTransformer
except ImportError:
    from model import ModularArithmeticTransformer  # type: ignore


CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")


@dataclass
class MatrixRankPoint:
    step: int
    name: str
    shape: Tuple[int, ...]
    effective_rank: float
    stable_rank: float
    singular_values: List[float]  # truncated (top 32) for storage


def list_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    for p in run_dir.iterdir():
        m = CHECKPOINT_RE.match(p.name)
        if m:
            items.append((int(m.group(1)), p))
    return sorted(items)


def effective_rank(s: torch.Tensor) -> float:
    s = s.float().clamp(min=0.0)
    total = s.sum()
    if total <= 1e-12:
        return 0.0
    p = s / total
    p = p.clamp(min=1e-30)
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


def stable_rank(s: torch.Tensor) -> float:
    s = s.float()
    smax = s.max()
    if smax <= 1e-12:
        return 0.0
    return float((s.pow(2).sum() / smax.pow(2)).item())


def is_2d_param(name: str, t: torch.Tensor) -> bool:
    if t.ndim != 2:
        return False
    if t.numel() < 4:
        return False
    return True


def matrix_spectra(
    state_dict: Dict[str, torch.Tensor], step: int
) -> List[MatrixRankPoint]:
    points: List[MatrixRankPoint] = []
    for name, tensor in state_dict.items():
        if not is_2d_param(name, tensor):
            continue
        with torch.no_grad():
            s = torch.linalg.svdvals(tensor.detach().to("cpu", dtype=torch.float32))
        er = effective_rank(s)
        sr = stable_rank(s)
        points.append(MatrixRankPoint(
            step=step,
            name=name,
            shape=tuple(tensor.shape),
            effective_rank=er,
            stable_rank=sr,
            singular_values=[float(x) for x in s[: min(32, s.numel())].tolist()],
        ))
    return points


def load_run_config(run_dir: Path) -> Optional[dict]:
    res = run_dir / "results.json"
    if not res.exists():
        return None
    with res.open() as f:
        data = json.load(f)
    return data


def analyze_run(run_dir: Path) -> Dict:
    """Return dict of {step: [MatrixRankPoint, ...]} plus run metadata."""
    ckpts = list_checkpoints(run_dir)
    if not ckpts:
        raise SystemExit(f"no checkpoints found in {run_dir}")
    print(f"[run] {run_dir}: {len(ckpts)} checkpoints, "
          f"steps {ckpts[0][0]}..{ckpts[-1][0]}")
    series: Dict[int, List[MatrixRankPoint]] = {}
    for step, path in ckpts:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            sd = ckpt["model_state"]
        else:
            sd = ckpt  # raw state_dict fallback
        series[step] = matrix_spectra(sd, step)
    meta = load_run_config(run_dir) or {}
    return {
        "run_dir": str(run_dir),
        "config": meta.get("config", {}),
        "grokked": meta.get("grokked"),
        "grokking_step": meta.get("grokking_step"),
        "final_test_acc": meta.get("final_test_acc"),
        "history": meta.get("history", []),
        "series": series,
    }


def to_per_matrix_traces(series: Dict[int, List[MatrixRankPoint]]):
    steps = sorted(series.keys())
    names = sorted({p.name for ps in series.values() for p in ps})
    ranks: Dict[str, List[float]] = {n: [] for n in names}
    stable: Dict[str, List[float]] = {n: [] for n in names}
    for s in steps:
        by_name = {p.name: p for p in series[s]}
        for n in names:
            if n in by_name:
                ranks[n].append(by_name[n].effective_rank)
                stable[n].append(by_name[n].stable_rank)
            else:
                ranks[n].append(float("nan"))
                stable[n].append(float("nan"))
    return steps, names, ranks, stable


def find_sharpest_transition(steps, ranks):
    """For each matrix, return (max_jump, step_after, normalized_jump)."""
    out = {}
    for name, vals in ranks.items():
        arr = np.array(vals, dtype=float)
        if len(arr) < 2 or np.all(np.isnan(arr)):
            out[name] = {"max_jump": float("nan"),
                         "step_after": None,
                         "rel_jump": float("nan")}
            continue
        diffs = np.diff(arr)
        if np.all(np.isnan(diffs)):
            out[name] = {"max_jump": float("nan"),
                         "step_after": None,
                         "rel_jump": float("nan")}
            continue
        idx = int(np.nanargmax(np.abs(diffs)))
        max_jump = float(diffs[idx])
        baseline = max(abs(arr[idx]), 1e-6)
        out[name] = {
            "max_jump": max_jump,
            "step_after": int(steps[idx + 1]),
            "rel_jump": max_jump / baseline,
        }
    return out


def plot_run_traces(steps, names, ranks, stable, out_path,
                    title, grokking_step=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharex=True)
    palette = plt.cm.tab20(np.linspace(0, 1, max(20, len(names))))
    for name, color in zip(names, palette):
        axes[0].plot(steps, ranks[name], label=name, color=color, marker="o",
                     markersize=3, linewidth=1.2)
        axes[1].plot(steps, stable[name], label=name, color=color, marker="o",
                     markersize=3, linewidth=1.2)
    if grokking_step is not None:
        for ax in axes:
            ax.axvline(grokking_step, color="red", linestyle="--",
                       alpha=0.6, label="grokking_step")
    axes[0].set_ylabel("effective rank exp(-sum p log p)")
    axes[0].set_title(f"{title}: effective rank")
    axes[1].set_ylabel("stable rank ||W||_F^2 / ||W||_2^2")
    axes[1].set_title(f"{title}: stable rank")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_compare(runs, out_path):
    """One panel per matrix; one line per run; show effective rank vs step."""
    all_names = sorted({n for run in runs
                        for n in to_per_matrix_traces(run["series"])[1]})
    n_mats = len(all_names)
    cols = 3
    rows = math.ceil(n_mats / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows),
                             squeeze=False)
    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(runs))))
    for idx, name in enumerate(all_names):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        for run, color in zip(runs, palette):
            steps, names, ranks, _ = to_per_matrix_traces(run["series"])
            label = (
                f"wd={run['config'].get('weight_decay')}"
                f" noise={run['config'].get('noise_fraction')}"
                f" seed={run['config'].get('seed')}"
            )
            if name in ranks:
                ax.plot(steps, ranks[name], marker="o", color=color, label=label,
                        markersize=3, linewidth=1.2)
            if run.get("grokking_step") is not None:
                ax.axvline(run["grokking_step"], color=color, linestyle="--",
                           alpha=0.4)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("eff. rank")
        ax.grid(alpha=0.3)
    # Hide any extra empty subplots
    for k in range(n_mats, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(handles), 4), fontsize=9)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_run_report(run, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    steps, names, ranks, stable = to_per_matrix_traces(run["series"])
    transitions = find_sharpest_transition(steps, ranks)
    cfg = run["config"]
    tag = (
        f"wd{cfg.get('weight_decay', '?')}_noise{cfg.get('noise_fraction', '?')}"
        f"_seed{cfg.get('seed', '?')}"
    ).replace(".", "p")
    plot_run_traces(
        steps, names, ranks, stable,
        out_dir / f"trace_{tag}.png",
        title=f"wd={cfg.get('weight_decay')} noise={cfg.get('noise_fraction')} seed={cfg.get('seed')}",
        grokking_step=run.get("grokking_step"),
    )
    # Per-run JSON
    serial = {
        "run_dir": run["run_dir"],
        "config": cfg,
        "grokked": run["grokked"],
        "grokking_step": run["grokking_step"],
        "final_test_acc": run["final_test_acc"],
        "steps": steps,
        "matrix_names": names,
        "effective_rank": {n: list(map(float, ranks[n])) for n in names},
        "stable_rank": {n: list(map(float, stable[n])) for n in names},
        "transitions": transitions,
    }
    out_path = out_dir / f"trace_{tag}.json"
    with out_path.open("w") as f:
        json.dump(serial, f, indent=2, default=lambda o: None)
    return tag, transitions


def write_overall_report(runs, transitions_per_run, out_dir: Path):
    lines = []
    lines.append("# Causal circuit rescue — per-matrix rank trajectories\n\n")
    lines.append(
        "For each run we recorded the effective rank "
        "(`exp(-sum p log p)` over normalised singular values) of every "
        "2D weight matrix at each saved checkpoint. The matrix with the "
        "largest single-step jump is the candidate 'circuit that grokking "
        "uses'.\n\n"
    )
    lines.append("| run | weight_decay | noise | seed | grokked | grok_step | top circuit | jump_step | rel_jump |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for run, (tag, transitions) in zip(runs, transitions_per_run):
        cfg = run["config"]
        if not transitions:
            continue
        # find matrix with largest |rel_jump|
        best_name = max(transitions,
                        key=lambda n: abs(transitions[n].get("rel_jump") or 0.0))
        t = transitions[best_name]
        lines.append(
            f"| {tag} | {cfg.get('weight_decay')} | {cfg.get('noise_fraction')} "
            f"| {cfg.get('seed')} | {run.get('grokked')} | "
            f"{run.get('grokking_step')} | `{best_name}` | "
            f"{t.get('step_after')} | {(t.get('rel_jump') or 0):+.3f} |\n"
        )
    lines.append("\n## Per-matrix top jump across all runs\n\n")
    lines.append("Aggregated: average rel_jump per matrix, sorted descending.\n\n")
    matrix_names = sorted({n for _, t in transitions_per_run for n in t})
    avg_rel_jump = []
    for n in matrix_names:
        vals = [t[n].get("rel_jump") for _, t in transitions_per_run
                if n in t and t[n].get("rel_jump") is not None]
        if vals:
            avg_rel_jump.append((n, float(np.mean(vals)), len(vals)))
    avg_rel_jump.sort(key=lambda x: abs(x[1]), reverse=True)
    lines.append("| matrix | mean(rel_jump) | n_runs |\n")
    lines.append("|---|---|---|\n")
    for n, v, k in avg_rel_jump:
        lines.append(f"| `{n}` | {v:+.3f} | {k} |\n")
    (out_dir / "causal_circuit_summary.md").write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+", type=Path, required=False,
        default=[
            Path("results/exp_c_grid/wd0.3/noise0/seed_42"),
            Path("results/exp_c_grid/wd0.3/noise0.05/seed_42"),
            Path("results/exp_c_grid/wd1/noise0.15/seed_42"),
            Path("results/exp_c_grid/wd1/noise0/seed_42"),
        ],
        help="Run directories containing checkpoint_*.pt and results.json",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("analysis/causal_circuit"),
        help="Where to write traces and plots",
    )
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    transitions_per_run = []
    for run_dir in args.runs:
        if not run_dir.exists():
            print(f"[skip] {run_dir} does not exist")
            continue
        run = analyze_run(run_dir)
        runs.append(run)
        tag, transitions = write_run_report(run, out_dir)
        transitions_per_run.append((tag, transitions))
        print(f"[run] wrote trace for {tag}")

    if not runs:
        raise SystemExit("no runs analyzed")

    # Cross-run comparison plot
    plot_compare(runs, out_dir / "compare_per_matrix.png")
    write_overall_report(runs, transitions_per_run, out_dir)
    print(f"[done] wrote {out_dir/'causal_circuit_summary.md'}")
    print(f"[done] wrote {out_dir/'compare_per_matrix.png'}")


if __name__ == "__main__":
    main()
