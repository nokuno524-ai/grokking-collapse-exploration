import os
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_capability_curves(results_dir: str = "results", output_dir: str = "analysis/figures"):
    """Plot capability emergence (Test Accuracy) curves."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

    results_path = Path(results_dir)
    if results_path.exists():
        for condition_dir in sorted(results_path.iterdir()):
            if not condition_dir.is_dir():
                continue

            results_file = condition_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)

                    history = data.get("history", [])
                    if history:
                        steps = [entry["step"] for entry in history]
                        accs = [entry["test_acc"] for entry in history]

                        color = colors.get(condition_dir.name, "gray")
                        ax.plot(steps, accs, label=condition_dir.name, color=color, linewidth=2)
                except Exception as e:
                    print(f"Error processing {results_file}: {e}")
    else:
        print(f"Results directory {results_dir} not found. Using mock data.")
        # Mock data for demonstration
        import numpy as np
        steps = list(range(0, 50000, 100))
        for name, color in colors.items():
            if name == "pure":
                accs = 1 / (1 + np.exp(-(np.array(steps) - 15000) / 1000))
            elif name == "low_collapse":
                accs = 1 / (1 + np.exp(-(np.array(steps) - 30000) / 2000))
            elif name == "medium_collapse":
                accs = 0.9 / (1 + np.exp(-(np.array(steps) - 45000) / 3000))
            else:
                accs = np.zeros(len(steps)) + np.random.uniform(0, 0.1, len(steps))
            ax.plot(steps, accs, label=name, color=color, linewidth=2)

    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label="Grokking Threshold")

    ax.set_title("Capability Emergence Curves (Test Accuracy)", fontsize=14)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "capability_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_capability_curves()
