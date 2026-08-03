import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

WDS = [0.0, 0.001, 0.01, 0.1, 1.0]
COLLAPSE_LEVELS = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

def plot_phase_diagram(results_dir: str = "results/wd_phase_diagram", output_dir: str = "results/phase_diagrams"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for wd in WDS:
        for condition in COLLAPSE_LEVELS:
            res_path = Path(results_dir) / f"wd_{wd}" / condition / "results.json"
            if res_path.exists():
                with open(res_path) as f:
                    data = json.load(f)

                grokked = 1.0 if data.get("grokked", False) else 0.0
                peak_acc = data.get("final_test_acc", 0.0)

                # Try to get peak accuracy from history if it's better
                history = data.get("history", [])
                if history:
                    peak_acc = max([h.get("test_acc", 0.0) for h in history] + [peak_acc])

                results.append({
                    "weight_decay": wd,
                    "collapse_severity": condition,
                    "grokked": grokked,
                    "peak_acc": peak_acc
                })

    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)

    # Plot Grokking Outcome Phase Diagram
    plt.figure(figsize=(10, 6))
    pivot_grok = df.pivot(index="weight_decay", columns="collapse_severity", values="grokked")
    # Reorder columns
    pivot_grok = pivot_grok[COLLAPSE_LEVELS]
    sns.heatmap(pivot_grok, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Grokking Outcome (1 = Grokked, 0 = No Grok)")
    plt.ylabel("Weight Decay")
    plt.xlabel("Collapse Severity")
    # Reverse y axis so smaller wd is at bottom
    plt.gca().invert_yaxis()
    plt.savefig(Path(output_dir) / "phase_diagram_grokking.png")
    plt.close()

    # Plot Peak Accuracy Phase Diagram
    plt.figure(figsize=(10, 6))
    pivot_acc = df.pivot(index="weight_decay", columns="collapse_severity", values="peak_acc")
    pivot_acc = pivot_acc[COLLAPSE_LEVELS]
    sns.heatmap(pivot_acc, annot=True, cmap="viridis", vmin=0, vmax=1)
    plt.title("Peak Test Accuracy")
    plt.ylabel("Weight Decay")
    plt.xlabel("Collapse Severity")
    plt.gca().invert_yaxis()
    plt.savefig(Path(output_dir) / "phase_diagram_peak_acc.png")
    plt.close()

    print(f"Phase diagrams saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/wd_phase_diagram")
    parser.add_argument("--output-dir", type=str, default="results/phase_diagrams")
    args = parser.parse_args()
    plot_phase_diagram(args.results_dir, args.output_dir)
