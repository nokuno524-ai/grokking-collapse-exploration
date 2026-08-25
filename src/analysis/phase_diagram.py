import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from collections import defaultdict
import json
import re

def classify_phase(train_acc, test_acc):
    """
    Classify the phase based on generalization gap (train_acc - test_acc).
    Threshold rule:
      gap < 0.1: 'grokked' (1)
      0.1 <= gap <= 0.9: 'transitioning' (0)
      gap > 0.9: 'memorizing-only' (-1)
    """
    gap = train_acc - test_acc
    if gap < 0.1:
        return 1
    elif gap <= 0.9:
        return 0
    else:
        return -1

def aggregate_metrics(results_list):
    """
    Given a list of dictionaries with 'severity', 'history' (list of dicts with 'step', 'train_acc', 'test_acc'),
    aggregates metrics into a 2D matrix of shape (len(steps), len(severities)).

    Returns:
        matrix: 2D numpy float array with NaNs for missing cells. Values are mean phase class across seeds.
        steps: sorted list of steps.
        severities: sorted list of severities.
    """
    # Group by (severity, step) -> list of phases
    data = defaultdict(list)
    steps_set = set()
    severities_set = set()

    for res in results_list:
        severity = res.get('severity')
        if severity is None:
            continue
        severities_set.add(severity)

        history = res.get('history', [])
        for entry in history:
            step = entry.get('step')
            train_acc = entry.get('train_acc')
            test_acc = entry.get('test_acc')

            if step is not None and train_acc is not None and test_acc is not None:
                steps_set.add(step)
                phase = classify_phase(train_acc, test_acc)
                data[(severity, step)].append(phase)

    steps = sorted(list(steps_set))
    severities = sorted(list(severities_set))

    matrix = np.full((len(steps), len(severities)), np.nan)

    for j, sev in enumerate(severities):
        for i, stp in enumerate(steps):
            phases = data.get((sev, stp), [])
            if phases:
                matrix[i, j] = np.mean(phases)

    return matrix, steps, severities

def plot_phase_diagram_2d(matrix, steps, severities, out_path):
    """
    Creates a 2D phase diagram. X-axis severity, Y-axis training step.
    Color cells by phase. Handles missing data by masking (NaNs).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # We map mean values [-1, 1]. Let's use a colormap that reflects this.
    # -1 (memorizing) -> Red
    # 0 (transitioning) -> Yellow
    # 1 (grokked) -> Green
    cmap = matplotlib.cm.get_cmap("RdYlGn")
    cmap.set_bad(color='lightgray') # Missing data

    # We want the y-axis to go from step 0 at the bottom to max step at the top.
    # matrix has shape (len(steps), len(severities)), so row 0 is steps[0]
    # imshow by default puts row 0 at the top, so we set origin='lower'

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, origin='lower', vmin=-1, vmax=1)

    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([f"{s:.2f}" for s in severities])

    # For steps, showing all might be too crowded. Show a subset.
    num_steps = len(steps)
    step_ticks = np.linspace(0, num_steps - 1, min(10, num_steps)).astype(int)
    ax.set_yticks(step_ticks)
    ax.set_yticklabels([str(steps[i]) for i in step_ticks])

    ax.set_xlabel("Collapse Severity")
    ax.set_ylabel("Training Step")
    ax.set_title("Phase Diagram: Grokking vs Collapse Severity")

    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['Memorizing', 'Transitioning', 'Grokked'])

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def plot_critical_step_vs_severity(matrix, steps, severities, out_path):
    """
    Line plot of severity vs. the step where the generalization gap first crosses the 'grokked' threshold.
    """
    critical_steps = []

    for j, sev in enumerate(severities):
        # Find the first step i where matrix[i, j] >= 0.5 (meaning majority seeds grokked)
        # Actually, let's say >= 0.5 is grokked, but we can also just check when it hits 1.0.
        # Let's say if mean phase > 0.5 it's grokked.
        grokked_idx = np.where(matrix[:, j] >= 0.5)[0]
        if len(grokked_idx) > 0:
            critical_steps.append(steps[grokked_idx[0]])
        else:
            critical_steps.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))

    valid_mask = ~np.isnan(critical_steps)
    valid_sev = np.array(severities)[valid_mask]
    valid_crit = np.array(critical_steps)[valid_mask]

    ax.plot(valid_sev, valid_crit, marker='o', linestyle='-', color='b')
    ax.set_xlabel("Collapse Severity")
    ax.set_ylabel("Critical Step (Grokking)")
    ax.set_title("Critical Grokking Step vs Collapse Severity")

    # Set y limits nicely
    if len(valid_crit) > 0:
        ax.set_ylim(0, max(valid_crit) * 1.1)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    GRID_ROOT = PROJECT_ROOT / "results" / "grid"
    OUT_DIR = PROJECT_ROOT / "analysis"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gather results
    DIR_RE = re.compile(r"^level(?P<level>[\d.]+)_sev(?P<sev>[\d.]+)$")
    SEED_RE = re.compile(r"^seed_(?P<seed>\d+)$")

    results_list = []

    # We will just look at one level for a clean 2D plot, say level=0.15, or aggregate across levels?
    # Or just use the exp_c_grid which has wd and noise.
    # Let's read exp_c_grid since it has noise which is collapse severity equivalent.
    EXP_C_GRID = PROJECT_ROOT / "results" / "exp_c_grid"

    # Let's use exp_c_grid for wd=0.3
    if EXP_C_GRID.exists():
        wd_dir = EXP_C_GRID / "wd0.3"
        if wd_dir.exists():
            for noise_dir in wd_dir.iterdir():
                if not noise_dir.is_dir(): continue
                if not noise_dir.name.startswith("noise"): continue
                noise_val = float(noise_dir.name.replace("noise", ""))

                for seed_dir in noise_dir.iterdir():
                    if not seed_dir.is_dir(): continue
                    if not seed_dir.name.startswith("seed_"): continue

                    res_path = seed_dir / "results.json"
                    if res_path.exists():
                        try:
                            with res_path.open() as f:
                                data = json.load(f)
                            # map noise to severity for our pipeline
                            data['severity'] = noise_val
                            results_list.append(data)
                        except Exception:
                            pass

    if not results_list:
        print("No data found!")
        return

    matrix, steps, severities = aggregate_metrics(results_list)

    plot_phase_diagram_2d(matrix, steps, severities, OUT_DIR / "phase_diagram_2d.png")
    plot_critical_step_vs_severity(matrix, steps, severities, OUT_DIR / "critical_step_vs_severity.png")
    print("Saved phase_diagram_2d.png and critical_step_vs_severity.png to", OUT_DIR)

if __name__ == "__main__":
    main()
