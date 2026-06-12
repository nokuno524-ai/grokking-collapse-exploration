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

def plot_phase_diagram(data: Dict[str, Dict[str, Any]], output_path: str = "visualizations/phase_diagram.png"):
    """Plots a 2D map of collapse severity vs grokking accuracy/probability."""
    if not data:
        # Empty plot to satisfy tests if no data
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel("Collapse Severity")
        ax.set_ylabel("Final Test Accuracy")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Plotted empty figure (no data).")
        return

    severities = []
    accuracies = []

    for cond, info in data.items():
        if "config" in info and "collapse_severity" in info["config"]:
            if "final_test_acc" in info:
                severities.append(info["config"]["collapse_severity"])
                accuracies.append(info["final_test_acc"])

    if not severities:
        print("No valid severity/accuracy data found.")
        return

    # Sort data for plotting lines correctly
    sorted_pairs = sorted(zip(severities, accuracies))
    severities, accuracies = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, 6))

    # We plot the raw points
    ax.scatter(severities, accuracies, color='black', zorder=5, s=50, label='Experiment Runs')

    # We also plot a line connecting them
    ax.plot(severities, accuracies, color='#3498db', alpha=0.7, linestyle='--', zorder=4)

    # Highlight the grokking threshold
    ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.8, label="Grokking Threshold")

    ax.set_title("Grokking Phase Diagram")
    ax.set_xlabel("Collapse Severity (Fraction of Fake Data)")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_ylim(-0.05, 1.05)

    ax.legend(loc="lower left")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved phase diagram to {output_path}")

if __name__ == "__main__":
    data = load_results("results")
    plot_phase_diagram(data)
