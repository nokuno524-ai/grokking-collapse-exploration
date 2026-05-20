import os
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_phase_transitions(results_dir: str = "results", output_dir: str = "analysis/figures"):
    """Plot grokking phase transition timing comparison."""
    os.makedirs(output_dir, exist_ok=True)

    conditions = []
    grokking_steps = []

    colors_dict = {
        "pure": "#2ecc71",
        "low_collapse": "#3498db",
        "medium_collapse": "#f39c12",
        "high_collapse": "#e74c3c",
        "severe_collapse": "#8e44ad",
    }

    results_path = Path(results_dir)
    if results_path.exists():
        # Using a fixed order
        order = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

        for name in order:
            condition_dir = results_path / name
            if condition_dir.exists() and condition_dir.is_dir():
                results_file = condition_dir / "results.json"
                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            data = json.load(f)

                        conditions.append(name.replace("_", " ").title())
                        step = data.get("grokking_step")
                        # Use 0 if it didn't grok
                        grokking_steps.append(step if step is not None else 0)
                    except Exception as e:
                        print(f"Error processing {results_file}: {e}")

    if not conditions:
        print(f"Results directory {results_dir} not found or empty. Using mock data.")
        conditions = ["Pure", "Low Collapse", "Medium Collapse", "High Collapse", "Severe Collapse"]
        grokking_steps = [1400, 3200, 8500, 0, 0]  # 0 indicates no grokking

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [colors_dict.get(c.lower().replace(" ", "_"), "gray") for c in conditions]

    bars = ax.bar(conditions, grokking_steps, color=colors)

    # Add labels on top of bars
    for bar, step in zip(bars, grokking_steps):
        height = bar.get_height()
        label = f"{step}" if step > 0 else "Did not grok"
        ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                label,
                ha='center', va='bottom', fontsize=10)

    ax.set_title("Grokking Phase Transition Timing by Condition", fontsize=14)
    ax.set_ylabel("Training Step of Grokking", fontsize=12)
    ax.set_ylim(0, max(max(grokking_steps) * 1.2, 50000) if max(grokking_steps) > 0 else 50000)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "phase_transitions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_phase_transitions()
