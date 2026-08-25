"""
Plots generated sweep results.
Reads a CSV of aggregated run metrics and produces scatter plots
with mean trend lines for grokking steps and final accuracy.
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot sweep results without pandas.")
    parser.add_argument("--results-csv", required=True, help="Input CSV file from sweep/collect.py.")
    parser.add_argument("--x-axis", required=True, help="Column name to use for the X-axis (e.g. collapse_severity).")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots.")
    parser.add_argument("--log-x", action="store_true", help="Use log scale for X-axis.")
    parser.add_argument("--log-y", action="store_true", help="Use log scale for Y-axis (grok_step).")
    return parser.parse_args()

def safe_float(val):
    """Safely convert a string to float, returning NaN if invalid."""
    try:
        if val is None or str(val).strip().lower() in ('', 'nan', 'none'):
            return float('nan')
        return float(val)
    except ValueError:
        return float('nan')

def plot_results():
    """Main execution to read results CSV, group metrics, and generate plots."""
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_vals_all = []
    grok_all = []
    acc_all = []

    # Read the CSV and filter COMPLETE runs
    with open(args.results_csv, "r") as f:
        reader = csv.DictReader(f)
        if args.x_axis not in reader.fieldnames:
            print(f"Error: {args.x_axis} not found in CSV headers.")
            return

        for row in reader:
            if row.get("status") == "COMPLETE":
                x_val = safe_float(row.get(args.x_axis))
                grok = safe_float(row.get("grok_step"))
                acc = safe_float(row.get("final_acc"))

                if not np.isnan(x_val):
                    x_vals_all.append(x_val)
                    grok_all.append(grok)
                    acc_all.append(acc)

    if not x_vals_all:
        print("No COMPLETE runs found to plot.")
        return

    x_arr = np.array(x_vals_all)
    grok_arr = np.array(grok_all)
    acc_arr = np.array(acc_all)

    # Sort by X axis for consistent plotting
    sort_idx = np.argsort(x_arr)
    x_arr = x_arr[sort_idx]
    grok_arr = grok_arr[sort_idx]
    acc_arr = acc_arr[sort_idx]

    # Group by x-axis to compute means
    grouped_grok = defaultdict(list)
    grouped_acc = defaultdict(list)

    for x, g, a in zip(x_arr, grok_arr, acc_arr):
        if not np.isnan(g):
            grouped_grok[x].append(g)
        if not np.isnan(a):
            grouped_acc[x].append(a)

    unique_x_grok = sorted(grouped_grok.keys())
    mean_grok = [np.mean(grouped_grok[x]) for x in unique_x_grok]

    unique_x_acc = sorted(grouped_acc.keys())
    mean_acc = [np.mean(grouped_acc[x]) for x in unique_x_acc]

    # Plot 1: Grok Step vs X
    plt.figure(figsize=(8, 6))

    # Scatter all seeds
    plt.scatter(x_arr, grok_arr, alpha=0.6, label="Seeds")

    # Plot mean line
    if unique_x_grok:
        plt.plot(unique_x_grok, mean_grok, 'r-', linewidth=2, label="Mean")

    if args.log_x:
        plt.xscale("log")
    if args.log_y:
        plt.yscale("log")

    plt.xlabel(args.x_axis)
    plt.ylabel("Grokking Step")
    plt.title(f"Grokking Step vs {args.x_axis}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_file = out_dir / f"grok_vs_{args.x_axis}.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved {out_file}")
    plt.close()

    # Plot 2: Final Accuracy vs X
    plt.figure(figsize=(8, 6))

    plt.scatter(x_arr, acc_arr, alpha=0.6, label="Seeds")

    if unique_x_acc:
        plt.plot(unique_x_acc, mean_acc, 'b-', linewidth=2, label="Mean")

    if args.log_x:
        plt.xscale("log")

    plt.xlabel(args.x_axis)
    plt.ylabel("Final Test Accuracy")
    plt.title(f"Final Accuracy vs {args.x_axis}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_file_acc = out_dir / f"acc_vs_{args.x_axis}.png"
    plt.savefig(out_file_acc, dpi=300, bbox_inches='tight')
    print(f"Saved {out_file_acc}")
    plt.close()

if __name__ == "__main__":
    plot_results()
