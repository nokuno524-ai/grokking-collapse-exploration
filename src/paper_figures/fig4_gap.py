import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures.plot_utils import set_style, get_color_palette

def generate_gap_figure(registry_path: Path, output_dir: Path):
    set_style()

    with open(registry_path) as f:
        registry = json.load(f)

    target_conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = get_color_palette()

    has_labels = False
    for condition in target_conditions:
        runs = [r for r in registry if r.get("condition_name") == condition and "seed_sweep" in r.get("run_path", "")]
        if not runs:
            runs = [r for r in registry if r.get("condition_name") == condition]

        if not runs:
            continue

        run = next((r for r in runs if r.get("seed") == 42), runs[0])
        run_path = Path(run["run_path"]) / "results.json"

        try:
            with open(run_path) as f:
                data = json.load(f)

            history = data.get("history", [])
            steps = [h["step"] for h in history]
            train_loss = np.array([h["train_loss"] for h in history])
            test_loss = np.array([h["test_loss"] for h in history])

            # Loss gap
            gap = test_loss - train_loss

            label = condition.replace("_", " ").title()
            ax.plot(steps, gap, color=colors[condition], label=label, linewidth=1.5)
            has_labels = True
        except Exception as e:
            print(f"Error loading {run_path}: {e}")

    ax.set_title("Generalization Gap (Test Loss - Train Loss)")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss Gap")

    if has_labels:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    out_path = output_dir / "fig4_gap.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_gap_figure(Path("results/registry.json"), Path("paper/figures"))
