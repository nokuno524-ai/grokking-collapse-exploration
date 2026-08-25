import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_metrics_grid(results_dir: Path) -> Dict[float, Dict[int, List[Tuple[float, float]]]]:
    """
    Scans results_dir for results.json files and aggregates test_acc and train_acc
    grouped by collapse_severity and step.

    Returns:
        Dict mapping severity to Dict mapping step to List of (train_acc, test_acc) tuples.
    """
    data: Dict[float, Dict[int, List[Tuple[float, float]]]] = {}
    for p in results_dir.rglob("results.json"):
        try:
            with open(p, "r") as f:
                d = json.load(f)
        except Exception:
            continue

        cfg = d.get("config", {})
        sev = cfg.get("collapse_severity")
        if sev is None:
            continue

        sev = float(sev)
        if sev not in data:
            data[sev] = {}

        history = d.get("history", [])
        for h in history:
            step = h.get("step")
            train_acc = h.get("train_acc")
            test_acc = h.get("test_acc")
            if step is not None and train_acc is not None and test_acc is not None:
                if step not in data[sev]:
                    data[sev][step] = []
                data[sev][step].append((float(train_acc), float(test_acc)))

    return data

def aggregate_metrics(
    data: Dict[float, Dict[int, List[Tuple[float, float]]]]
) -> Dict[float, Dict[int, Tuple[float, float]]]:
    """
    Averages the train and test accuracies across all seeds for each cell.

    Returns:
        Dict mapping severity to Dict mapping step to (mean_train_acc, mean_test_acc).
    """
    agg = {}
    for sev, steps_data in data.items():
        agg[sev] = {}
        for step, acc_list in steps_data.items():
            if not acc_list:
                continue
            mean_train = sum(a[0] for a in acc_list) / len(acc_list)
            mean_test = sum(a[1] for a in acc_list) / len(acc_list)
            agg[sev][step] = (mean_train, mean_test)
    return agg

def classify_phase(train_acc: float, test_acc: float, gap_threshold: float = 0.05, memorization_train_threshold: float = 0.95) -> int:
    """
    Classify the training phase based on a generalization gap rule.
    Returns:
      2: Grokked (train_acc >= memorization_train_threshold AND train_acc - test_acc <= gap_threshold)
      0: Memorizing-only (train_acc >= memorization_train_threshold AND train_acc - test_acc > gap_threshold)
      1: Transitioning / Not memorizing (train_acc < memorization_train_threshold)
    """
    if train_acc < memorization_train_threshold:
        return 1

    gap = train_acc - test_acc
    if gap <= gap_threshold:
        return 2
    else:
        return 0

def build_phase_matrix(
    agg_data: Dict[float, Dict[int, Tuple[float, float]]],
    severities: List[float],
    steps: List[int],
    gap_threshold: float = 0.05
) -> np.ndarray:
    """
    Build a 2D numpy array of phases (steps x severities) for plotting.
    Missing cells are filled with np.nan.
    """
    matrix = np.full((len(steps), len(severities)), np.nan, dtype=np.float32)
    for j, sev in enumerate(severities):
        if sev not in agg_data:
            continue
        for i, step in enumerate(steps):
            if step in agg_data[sev]:
                train_acc, test_acc = agg_data[sev][step]
                matrix[i, j] = classify_phase(train_acc, test_acc, gap_threshold)
    return matrix

def find_critical_steps(
    agg_data: Dict[float, Dict[int, Tuple[float, float]]],
    severities: List[float],
    steps: List[int],
    gap_threshold: float = 0.05
) -> List[float]:
    """
    Find the first step (critical step) where the model has grokked for each severity.
    If no such step exists, or data is missing, returns np.nan for that severity.
    """
    critical_steps = []
    for sev in severities:
        c_step = np.nan
        if sev in agg_data:
            for step in sorted(steps):
                if step in agg_data[sev]:
                    train_acc, test_acc = agg_data[sev][step]
                    phase = classify_phase(train_acc, test_acc, gap_threshold)
                    if phase == 2:
                        c_step = float(step)
                        break
        critical_steps.append(c_step)
    return critical_steps

def plot_phase_diagram(
    matrix: np.ndarray,
    severities: List[float],
    steps: List[int],
    output_path: Path
):
    """
    Plots the phase diagram as a 2D heatmap.
    Saves to output_path without calling plt.show().
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Custom colormap to handle NaNs (e.g., grey for missing)
    cmap = matplotlib.colors.ListedColormap(['#d73027', '#fee090', '#4575b4']) # Red: Mem, Yellow: Trans, Blue: Grokked
    cmap.set_bad(color='grey', alpha=0.5)

    # 0 -> 0, 1 -> 1, 2 -> 2. The data contains 0, 1, 2, and NaN.
    # Set bounds so 0 maps to color 0, 1 to color 1, 2 to color 2
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(
        matrix,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        norm=norm,
        extent=[-0.5, len(severities)-0.5, steps[0] if len(steps)>0 else 0, steps[-1] if len(steps)>0 else 0]
    )

    ax.set_xticks(np.arange(len(severities)))
    ax.set_xticklabels([f"{s:.2f}" for s in severities])
    ax.set_xlabel("Collapse Severity")
    ax.set_ylabel("Training Step")
    ax.set_title("Phase Diagram: Memorization vs Grokking")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Memorizing-only', 'Transitioning', 'Grokked'])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_critical_step(
    severities: List[float],
    critical_steps: List[float],
    output_path: Path
):
    """
    Plots the critical grokking step vs collapse severity.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(severities, critical_steps, marker='o', linestyle='-', color='indigo')

    ax.set_xlabel("Collapse Severity")
    ax.set_ylabel("Critical Grokking Step")
    ax.set_title("Critical Step vs. Severity")
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    results_dir = Path("results")
    print(f"Loading data from {results_dir}...")
    raw_data = load_metrics_grid(results_dir)
    agg_data = aggregate_metrics(raw_data)

    severities = sorted(list(agg_data.keys()))
    steps_set = set()
    for sev_data in agg_data.values():
        steps_set.update(sev_data.keys())
    steps = sorted(list(steps_set))

    print(f"Severities found: {severities}")
    print(f"Steps range: {steps[0] if steps else None} to {steps[-1] if steps else None}")

    matrix = build_phase_matrix(agg_data, severities, steps)
    c_steps = find_critical_steps(agg_data, severities, steps)

    out_dir = Path("analysis")
    plot_phase_diagram(matrix, severities, steps, out_dir / "phase_diagram.png")
    plot_critical_step(severities, c_steps, out_dir / "critical_step.png")
    print(f"Plots saved to {out_dir}")
