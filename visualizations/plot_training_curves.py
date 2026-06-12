import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List

matplotlib.use('Agg')

# Styling for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

SEVERITY_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
COLORS = {
    "pure": "#2ecc71",          # Green
    "low_collapse": "#3498db",  # Blue
    "medium_collapse": "#f39c12", # Orange
    "high_collapse": "#e74c3c",   # Red
    "severe_collapse": "#8e44ad"  # Purple
}
LABELS = {
    "pure": "Pure (No Collapse)",
    "low_collapse": "Low Severity",
    "medium_collapse": "Medium Severity",
    "high_collapse": "High Severity",
    "severe_collapse": "Severe Severity"
}


def load_results(results_dir: str = "results") -> Dict[str, Dict[str, Any]]:
    """Loads results.json from all subdirectories in results_dir."""
    base_path = Path(results_dir)
    data = {}
    if not base_path.exists():
        print(f"Warning: {results_dir} does not exist.")
        return data

    for p in base_path.iterdir():
        if p.is_dir():
            json_file = p / "results.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data[p.name] = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: could not parse {json_file}")
    return data

def extract_metric(history: List[Dict[str, Any]], metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract steps and metric values from history, handling edge cases."""
    if not history:
        return np.array([]), np.array([])

    steps = []
    values = []
    for entry in history:
        if "step" in entry and metric_name in entry:
            steps.append(entry["step"])
            values.append(entry[metric_name])

    return np.array(steps), np.array(values)

def plot_curves(data: Dict[str, Dict[str, Any]], output_path: str = "visualizations/training_curves.png"):
    """Plots training curves across different severity levels."""
    if not data:
        # Create an empty plot to satisfy tests if no data
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax in axes:
            ax.set_xlabel("Steps")
            ax.set_ylabel("No Data")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted empty figure (no data).")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Test Loss
    ax_loss = axes[0]
    ax_loss.set_title("Test Loss vs. Steps")
    ax_loss.set_xlabel("Steps")
    ax_loss.set_ylabel("Test Loss")
    ax_loss.set_yscale("log")

    # 2. Test Accuracy
    ax_acc = axes[1]
    ax_acc.set_title("Test Accuracy vs. Steps")
    ax_acc.set_xlabel("Steps")
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_ylim(-0.05, 1.05)

    # 3. Weight Norm
    ax_norm = axes[2]
    ax_norm.set_title("Weight Norm vs. Steps")
    ax_norm.set_xlabel("Steps")
    ax_norm.set_ylabel("L2 Weight Norm")

    # Order the conditions
    available_conditions = set(data.keys())
    ordered_conditions = [c for c in SEVERITY_ORDER if c in available_conditions]
    ordered_conditions.extend([c for c in available_conditions if c not in SEVERITY_ORDER])

    for cond in ordered_conditions:
        history = data[cond].get("history", [])
        color = COLORS.get(cond, "#7f8c8d")
        label = LABELS.get(cond, cond)

        # Loss
        steps, vals = extract_metric(history, "test_loss")
        if len(steps) > 0:
            ax_loss.plot(steps, vals, color=color, label=label, alpha=0.8)

        # Accuracy
        steps, vals = extract_metric(history, "test_acc")
        if len(steps) > 0:
            ax_acc.plot(steps, vals, color=color, label=label, alpha=0.8)

        # Weight norm
        steps, vals = extract_metric(history, "weight_norm")
        if len(steps) > 0:
            ax_norm.plot(steps, vals, color=color, label=label, alpha=0.8)

    for ax in axes:
        # Only add legend if there are labels
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    fig.tight_layout()
    # Save the plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved training curves to {output_path}")

if __name__ == "__main__":
    data = load_results("results")
    plot_curves(data)
