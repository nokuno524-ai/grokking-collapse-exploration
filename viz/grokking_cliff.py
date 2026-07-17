"""
Grokking Cliff (Phase Diagram) visualization.
Generates publication-quality phase diagrams from grid search results.
Shows test accuracy and grokking step across collapse level and severity.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

def parse_grid_results(grid_dir: Path) -> pd.DataFrame:
    """Parse grid results into a DataFrame."""
    data = []

    # Grid dir structure: levelX_sevY/seed_Z/results.json
    for level_dir in grid_dir.iterdir():
        if not level_dir.is_dir() or not level_dir.name.startswith("level"):
            continue

        # Parse level and severity from dir name
        # e.g., level0.15_sev0.6
        parts = level_dir.name.split("_")
        if len(parts) != 2:
            continue

        level = float(parts[0].replace("level", ""))
        severity = float(parts[1].replace("sev", ""))

        for seed_dir in level_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            seed = int(seed_dir.name.replace("seed_", ""))
            results_file = seed_dir / "results.json"

            if results_file.exists():
                with open(results_file) as f:
                    try:
                        res = json.load(f)
                        data.append({
                            "level": level,
                            "severity": severity,
                            "seed": seed,
                            "grokked": res.get("grokked", False),
                            "grokking_step": res.get("grokking_step", 50000) or 50000,
                            "test_acc": res.get("final_test_acc", 0.0),
                            "fourier": res.get("final_fourier_concentration", 0.0)
                        })
                    except Exception as e:
                        print(f"Error parsing {results_file}: {e}")

    return pd.DataFrame(data)

def plot_phase_diagram(df: pd.DataFrame, metric: str, title: str,
                       cmap: str, save_path: Path,
                       vmin: float = None, vmax: float = None):
    """Plot a phase diagram (heatmap) for a specific metric."""
    # Average across seeds
    pivot_df = df.pivot_table(index="severity", columns="level", values=metric, aggfunc="mean")

    # Sort indices
    pivot_df = pivot_df.sort_index(ascending=False) # higher severity on top
    pivot_df = pivot_df.sort_index(axis=1) # lower level on left

    plt.figure(figsize=(8, 6))

    # Setup styling
    sns.set_theme(style="white", context="paper", font_scale=1.5)

    # Plot heatmap
    ax = sns.heatmap(pivot_df, annot=True, fmt=".2f" if metric != "grokking_step" else ".0f",
                     cmap=cmap, vmin=vmin, vmax=vmax,
                     cbar_kws={'label': title},
                     linewidths=.5, square=True)

    ax.set_xlabel("Collapse Level (Contamination Fraction)")
    ax.set_ylabel("Collapse Severity (Temperature)")
    ax.set_title(f"Phase Diagram: {title}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

def plot_confidence_intervals(df: pd.DataFrame, save_path: Path):
    """Plot metric with confidence intervals across levels for different severities."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Test Accuracy
    sns.lineplot(data=df, x="level", y="test_acc", hue="severity",
                 marker="o", err_style="bars", ax=ax1, palette="viridis")
    ax1.set_title("Test Accuracy vs Collapse Level")
    ax1.set_xlabel("Collapse Level")
    ax1.set_ylabel("Final Test Accuracy")
    ax1.axhline(0.95, color='r', linestyle='--', alpha=0.5, label='Grokking Threshold')

    # Grokking Step
    sns.lineplot(data=df, x="level", y="grokking_step", hue="severity",
                 marker="s", err_style="bars", ax=ax2, palette="viridis")
    ax2.set_title("Grokking Delay vs Collapse Level")
    ax2.set_xlabel("Collapse Level")
    ax2.set_ylabel("Steps to Grok")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", type=str, default="results/grid")
    parser.add_argument("--output-dir", type=str, default="viz_output")
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if grid_dir.exists():
        print(f"Parsing grid results from {grid_dir}...")
        df = parse_grid_results(grid_dir)

        if not df.empty:
            print(f"Found {len(df)} runs. Generating phase diagrams...")

            # Test Accuracy Phase Diagram
            plot_phase_diagram(df, "test_acc", "Final Test Accuracy",
                               "RdYlGn", output_dir / "phase_diagram_acc.png",
                               vmin=0, vmax=1)

            # Grokking Step Phase Diagram
            # Inverse colormap so smaller steps (faster grokking) is greener
            plot_phase_diagram(df, "grokking_step", "Steps to Grok",
                               "RdYlGn_r", output_dir / "phase_diagram_steps.png")

            # Line plots with CI
            plot_confidence_intervals(df, output_dir / "collapse_scaling_ci.png")

            print("Done!")
        else:
            print("No valid results found in grid directory.")
    else:
        print(f"Grid directory {grid_dir} does not exist.")
