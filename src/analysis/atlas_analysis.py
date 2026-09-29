"""
Analysis tools for the transplant matrix output.
Generates portability heatmaps and summaries.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def analyze_transplant_matrix(json_path: Path, output_dir: Path):
    """
    Parses the transplant matrix JSON and produces heatmaps showing which
    components are condition-portable vs condition-locked.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    baselines = data["baselines"]
    results = data["results"]

    # Extract unique dimensions
    conditions = list(baselines.keys())
    components = list(set(r["component"] for r in results))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a separate heatmap for each component
    for component in components:
        # matrix[source_idx][target_idx]
        matrix = np.zeros((len(conditions), len(conditions)))

        for r in results:
            if r["component"] == component:
                s_idx = conditions.index(r["source"])
                t_idx = conditions.index(r["target"])
                # We plot the absolute accuracy of the patched model
                matrix[s_idx, t_idx] = r["test_acc"]

        plot_heatmap(
            matrix,
            conditions,
            f"Transplant Acc: {component}",
            output_dir / f"heatmap_{component}.png"
        )

    # Analyze portability
    # Portable: Donor X improves/maintains performance on Recipient Y (where X != Y).
    # Locked: Donor X only helps Recipient X.
    summary_lines = [
        "# Circuit Portability Analysis",
        "",
        "This report analyzes whether specific neural circuits (components) extracted from models trained under one data condition can successfully replace their counterparts in models trained under different data conditions (zero-shot transfer).",
        "",
        "## Component Portability Scores",
        "A portability score reflects how well a component from a *pure* donor maintains performance when placed into *collapsed* recipients, compared to the recipient's baseline.",
        "",
        "| Component | Portability (Pure -> Collapsed Avg Acc Diff) |",
        "|-----------|----------------------------------------------|"
    ]

    if "pure" in conditions:
        portability = {}
        for component in components:
            diffs = []
            for r in results:
                if r["component"] == component and r["source"] == "pure" and r["target"] != "pure":
                    diffs.append(r["acc_diff"])
            if diffs:
                portability[component] = sum(diffs) / len(diffs)

        for comp, score in sorted(portability.items(), key=lambda x: x[1], reverse=True):
            summary_lines.append(f"| {comp} | {score:+.3f} |")

        summary_lines.extend([
            "",
            "### Interpretation",
            "- **Condition-Portable (Positive/Near Zero Diff)**: Circuits that transfer well. A pure component dropped into a collapsed model restores/maintains grokked performance.",
            "- **Condition-Locked (Negative Diff)**: Circuits that fail to transfer. The pure component breaks the collapsed model, indicating the representation strategies are highly specialized to the data quality.",
        ])

    with open(output_dir / "PORTABILITY_SUMMARY.md", "w") as f:
        f.write("\n".join(summary_lines))


def plot_heatmap(matrix, labels, title, out_path):
    """
    Renders and saves a heatmap for a single transplant matrix.

    Args:
        matrix: 2D numpy array of test accuracies.
        labels: List of condition names for the axes.
        title: Title of the plot.
        out_path: Path where the plot will be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Recipient (Target)")
    ax.set_ylabel("Donor (Source)")
    ax.set_title(title)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Test Accuracy", rotation=-90, va="bottom")

    for i in range(len(labels)):
        for j in range(len(labels)):
            text_color = "black" if 0.3 < matrix[i, j] < 0.7 else "white"
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", color=text_color, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("analysis/transplant_matrix.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/transplant_atlas"))
    args = parser.parse_args()

    if args.input.exists():
        analyze_transplant_matrix(args.input, args.output_dir)
    else:
        print(f"File not found: {args.input}. Run sweep.py first.")
