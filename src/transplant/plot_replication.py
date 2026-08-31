import json
import argparse
from pathlib import Path

def plot_replication_forest(stats_json: Path, output_path: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not installed, skipping plot generation.")
        return

    with open(stats_json, "r") as f:
        data = json.load(f)

    if not data:
        print("[warn] No data found in stats json, skipping plot.")
        return

    conditions = list(set([r["condition"] for r in data]))
    components = list(set([r["component"] for r in data]))

    # Sort for consistency
    conditions.sort()
    components.sort()

    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 6), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for i, cond in enumerate(conditions):
        ax = axes[i]
        cond_data = [r for r in data if r["condition"] == cond]

        y_pos = np.arange(len(components))
        means = []
        errors_lower = []
        errors_upper = []
        colors = []

        for comp in components:
            row = next((r for r in cond_data if r["component"] == comp), None)
            if row:
                means.append(row["mean_diff"])
                errors_lower.append(row["mean_diff"] - row["ci_lower"])
                errors_upper.append(row["ci_upper"] - row["mean_diff"])
                colors.append("green" if row["replicates"] else "red")
            else:
                means.append(0)
                errors_lower.append(0)
                errors_upper.append(0)
                colors.append("gray")

        for m, y, el, eu, c in zip(means, y_pos, errors_lower, errors_upper, colors):
            ax.errorbar([m], [y], xerr=[[el], [eu]], fmt='o',
                       color='black', ecolor=c, capsize=4, elinewidth=2, markersize=6)

        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(components)
        ax.set_title(f"Condition: {cond}")
        ax.set_xlabel("Mean Accuracy Difference\n(Transplant - Baseline)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[info] Saved forest plot to {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-json", type=Path, default=Path("analysis/transplant_replication/replication_stats.json"))
    ap.add_argument("--output", type=Path, default=Path("analysis/transplant_replication/forest_plot.png"))
    args = ap.parse_args()

    plot_replication_forest(args.stats_json, args.output)

if __name__ == "__main__":
    main()
