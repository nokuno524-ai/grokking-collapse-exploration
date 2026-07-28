import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def aggregate_results(results_dir="results/scaling"):
    """
    Collect all results from scaling experiments.
    """
    root_path = Path(results_dir)
    data = []

    # Iterate through all results.json files
    for results_file in root_path.rglob("results.json"):
        try:
            with open(results_file, 'r') as f:
                res = json.load(f)

            config = res.get("config", {})

            entry = {
                "d_model": config.get("d_model"),
                "n_heads": config.get("n_heads"),
                "train_fraction": config.get("train_fraction"),
                "collapse_level": config.get("collapse_level"),
                "seed": config.get("seed"),
                "grokked": res.get("grokked", False),
                "grokking_step": res.get("grokking_step", np.nan),
                "final_test_acc": res.get("final_test_acc", 0.0),
                "final_train_acc": res.get("final_train_acc", 0.0),
                "final_weight_norm": res.get("final_weight_norm", 0.0),
                "final_embedding_rank": res.get("final_embedding_rank", 0.0),
                "final_fourier_concentration": res.get("final_fourier_concentration", 0.0)
            }
            data.append(entry)
        except Exception as e:
            print(f"Error parsing {results_file}: {e}")

    df = pd.DataFrame(data)
    return df

def generate_reports(df, output_dir="results/scaling_analysis"):
    """
    Generate summary tables, visualizations and a FINDINGS.md report.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        print("No data found to aggregate.")
        return

    # Generate CSV
    df.to_csv(out_path / "full_results.csv", index=False)

    # Generate summary tables
    summary_cols = ["grokked", "grokking_step", "final_test_acc"]

    # Group by model size and collapse level
    summary_df = df.groupby(["d_model", "collapse_level"])[summary_cols].mean().reset_index()
    summary_df.to_csv(out_path / "summary_by_model_collapse.csv", index=False)

    # Generate Heatmaps if possible
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 8))
        pivot_grok = df.pivot_table(
            values="grokking_step",
            index="d_model",
            columns="collapse_level",
            aggfunc=np.mean
        )
        sns.heatmap(pivot_grok, annot=True, fmt=".0f", cmap="viridis_r")
        plt.title("Mean Grokking Step by Model Size and Collapse Level")
        plt.savefig(out_path / "grokking_heatmap.png")
        plt.close()

        plt.figure(figsize=(10, 8))
        pivot_acc = df.pivot_table(
            values="final_test_acc",
            index="d_model",
            columns="collapse_level",
            aggfunc=np.mean
        )
        sns.heatmap(pivot_acc, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Final Test Accuracy by Model Size and Collapse Level")
        plt.savefig(out_path / "test_acc_heatmap.png")
        plt.close()

    # Produce FINDINGS.md
    with open(out_path / "FINDINGS.md", "w") as f:
        f.write("# Scaling Experiment Findings\n\n")
        f.write("## Overview\n")
        f.write(f"Total runs analyzed: {len(df)}\n\n")

        f.write("## Summary Metrics\n")
        f.write("### By Collapse Level\n")
        collapse_summary = df.groupby("collapse_level")["grokked"].mean().reset_index()
        collapse_summary.columns = ["Collapse Level", "Grokking Rate"]
        f.write(collapse_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("### By Model Size\n")
        model_summary = df.groupby("d_model")["grokked"].mean().reset_index()
        model_summary.columns = ["Model Size (d_model)", "Grokking Rate"]
        f.write(model_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Detailed Table\n")
        f.write(summary_df.to_markdown(index=False))

    print(f"Reports generated in {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/scaling")
    parser.add_argument("--output-dir", type=str, default="results/scaling_analysis")
    args = parser.parse_args()

    df = aggregate_results(args.results_dir)
    generate_reports(df, args.output_dir)

if __name__ == "__main__":
    main()
