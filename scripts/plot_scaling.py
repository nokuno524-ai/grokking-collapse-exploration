import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from typing import List, Dict, Any, Union

def get_onset_step(history: List[Dict[str, Any]], threshold: float = 0.9) -> float:
    """
    Finds the first step where test_acc >= threshold.
    Returns np.nan if not found.
    """
    for entry in history:
        if entry.get("test_acc", 0) >= threshold:
            return entry["step"]
    return np.nan

def plot_scaling_heatmaps(jsonl_path: Union[str, Path], out_dir: Union[str, Path]) -> None:
    """
    Generates heatmaps of grokking onset time vs model size and data size.

    Args:
        jsonl_path: Path to the JSONL results file from the scaling grid.
        out_dir: Directory to save the resulting heatmap images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read data
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if not data:
        print(f"No data found in {jsonl_path}")
        return

    # Extract unique values for axes
    # Sort model sizes logically rather than alphabetically if possible
    raw_sizes = list(set(d["model_size_name"] for d in data))
    size_order = {"tiny": 0, "small": 1, "base": 2, "large": 3}
    model_sizes = sorted(raw_sizes, key=lambda x: size_order.get(x, 99))
    data_fractions = sorted(list(set(d["train_fraction"] for d in data)))
    severities = sorted(list(set(d["condition_name"] for d in data)))

    # Process into heatmaps per severity
    for severity in severities:
        # Create a 2D grid: rows=model_size, cols=data_fraction
        grid = np.full((len(model_sizes), len(data_fractions)), np.nan)

        for d in data:
            if d["condition_name"] == severity:
                i = model_sizes.index(d["model_size_name"])
                j = data_fractions.index(d["train_fraction"])
                # Get grokking onset time
                onset = get_onset_step(d["history"], threshold=0.9)
                grid[i, j] = onset

        # Plot heatmap
        plt.figure(figsize=(8, 6))

        # Mask NaNs (runs that didn't grok)
        masked_grid = np.ma.masked_invalid(grid)

        cmap = plt.cm.viridis_r
        cmap.set_bad(color='red')  # Red for "never grokked"

        im = plt.imshow(masked_grid, cmap=cmap, aspect='auto', origin='lower')
        plt.colorbar(im, label="Grokking Onset Step")

        plt.xticks(np.arange(len(data_fractions)), data_fractions)
        plt.yticks(np.arange(len(model_sizes)), model_sizes)

        plt.xlabel("Train Data Fraction")
        plt.ylabel("Model Size")
        plt.title(f"Grokking Onset - {severity}\n(Red = Did not grok)")

        # Add text annotations
        for i in range(len(model_sizes)):
            for j in range(len(data_fractions)):
                val = grid[i, j]
                text = "Failed" if np.isnan(val) else f"{int(val)}"
                color = "white" if not np.isnan(val) and val < np.nanmax(grid)*0.5 else "black"
                if np.isnan(val):
                    color = "white"
                plt.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

        out_path = out_dir / f"scaling_heatmap_{severity}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling heatmaps")
    parser.add_argument("--in-file", type=str, default="results_scaling/scaling_results.jsonl")
    parser.add_argument("--out-dir", type=str, default="results_scaling/plots")
    args = parser.parse_args()

    plot_scaling_heatmaps(args.in_file, args.out_dir)
