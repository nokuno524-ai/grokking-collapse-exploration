import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.plot_utils import set_style, get_color_palette

def generate_grok_curves(registry_path: Path, output_dir: Path):
    set_style()

    with open(registry_path) as f:
        registry = json.load(f)

    # Filter for seed 42 to make clean plots, or average across seeds
    # Let's use the pure and collapse sweeps from multi_seed or seed_sweep

    # Target conditions we want to plot
    target_conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = get_color_palette()

    for condition in target_conditions:
        # Find runs matching this condition
        runs = [r for r in registry if r.get("condition_name") == condition and "seed_sweep" in r.get("run_path", "")]
        if not runs:
            # Fallback to general results
            runs = [r for r in registry if r.get("condition_name") == condition]

        if not runs:
            continue

        # Just pick one representative run per condition for clarity (e.g. seed 42)
        run = next((r for r in runs if r.get("seed") == 42), runs[0])
        run_path = Path(run["run_path"]) / "results.json"

        try:
            with open(run_path) as f:
                data = json.load(f)

            history = data.get("history", [])
            steps = [h["step"] for h in history]
            train_acc = [h["train_acc"] for h in history]
            test_acc = [h["test_acc"] for h in history]

            label = condition.replace("_", " ").title()
            axes[0].plot(steps, train_acc, color=colors[condition], label=label, linewidth=1.5)
            axes[1].plot(steps, test_acc, color=colors[condition], label=label, linewidth=1.5)
        except Exception as e:
            print(f"Error loading {run_path}: {e}")

    axes[0].set_title("Train Accuracy")
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Accuracy")

    axes[1].set_title("Test Accuracy (Grokking)")
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Accuracy")
    axes[1].axhline(0.95, color='gray', linestyle='--', alpha=0.8, label="95% Threshold")

    # Place legend outside
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    out_path = output_dir / "fig1_grok_curves.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_grok_curves(Path("results/registry.json"), Path("paper/figures"))
