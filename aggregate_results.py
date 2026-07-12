import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def find_result_files(base_dir):
    """Recursively finds all results.json files in a directory."""
    results_files = []
    for root, _, files in os.walk(base_dir):
        if "results.json" in files:
            results_files.append(os.path.join(root, "results.json"))
    return results_files

def aggregate_metrics(results_files):
    """Aggregates metrics from a list of results.json files grouping by config."""
    data = []

    for file_path in results_files:
        with open(file_path, "r") as f:
            try:
                result = json.load(f)

                # Extract key config parameters to group by
                config = result.get("config", {})
                data_config = config.get("data", {})
                training_config = config.get("training", {})

                collapse_ratio = data_config.get("collapse_ratio", 0.0)
                noise_fraction = data_config.get("noise_fraction", 0.0)
                weight_decay = training_config.get("weight_decay", 1.0)
                seed = training_config.get("seed", 42)

                # Extract metrics
                final_test_acc = result.get("final_test_acc", 0.0)
                final_weight_norm = result.get("final_weight_norm", 0.0)
                final_fourier_concentration = result.get("final_fourier_concentration", 0.0)
                grokking_step = result.get("grokking_step")

                # Treat missing grokking step as the max steps in the config
                if grokking_step is None:
                    grokking_step = float('inf') # Will handle inf later or exclude

                data.append({
                    "collapse_ratio": collapse_ratio,
                    "noise_fraction": noise_fraction,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "final_test_acc": final_test_acc,
                    "final_weight_norm": final_weight_norm,
                    "final_fourier_concentration": final_fourier_concentration,
                    "grokking_step": grokking_step,
                    "grokked": result.get("grokked", False)
                })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    return df

def generate_latex_table(df, group_by_cols, output_path):
    """Computes mean and std dev and outputs a LaTeX table."""
    if df.empty:
        print("No data to generate table.")
        return

    # Exclude non-grokked models from grokking step average, or just report % grokked
    agg_funcs = {
        "final_test_acc": ["mean", "std"],
        "final_weight_norm": ["mean", "std"],
        "final_fourier_concentration": ["mean", "std"],
        "grokked": ["mean"], # This will be the fraction of seeds that grokked
    }

    # We will compute grokking_step mean only for seeds that actually grokked
    grokking_stats = df[df["grokked"] == True].groupby(group_by_cols)["grokking_step"].agg(["mean", "std"]).reset_index()
    grokking_stats.rename(columns={"mean": "grokking_step_mean", "std": "grokking_step_std"}, inplace=True)

    stats = df.groupby(group_by_cols).agg(agg_funcs).reset_index()

    # Flatten multi-level columns
    stats.columns = ['_'.join(col).strip('_') if type(col) is tuple else col for col in stats.columns.values]

    # Merge grokking step stats back
    stats = pd.merge(stats, grokking_stats, on=group_by_cols, how="left")

    # Format the table
    # Constructing LaTeX string manually for better control over formatting

    latex_str = "\\begin{table*}[t]\n\\centering\n\\begin{tabular}{"
    latex_str += "l" * len(group_by_cols) + " | c c c c c}\n\\toprule\n"

    headers = [col.replace("_", "\\_").capitalize() for col in group_by_cols]
    headers += ["Test Acc", "Weight Norm", "Fourier Conc.", "Grokked Fraction", "Grok Step"]
    latex_str += " & ".join(headers) + " \\\\\n\\midrule\n"

    for _, row in stats.iterrows():
        row_str = []
        for col in group_by_cols:
            row_str.append(f"{row[col]:.2f}" if isinstance(row[col], float) else str(row[col]))

        test_acc = f"{row['final_test_acc_mean']:.2f} $\\pm$ {row['final_test_acc_std']:.2f}"
        weight_norm = f"{row['final_weight_norm_mean']:.1f} $\\pm$ {row['final_weight_norm_std']:.1f}"
        fourier = f"{row['final_fourier_concentration_mean']:.3f} $\\pm$ {row['final_fourier_concentration_std']:.3f}"
        grokked_frac = f"{row['grokked_mean']:.2f}"

        if pd.isna(row.get('grokking_step_mean')):
            grok_step = "-"
        else:
            grok_step = f"{row['grokking_step_mean']:.0f} $\\pm$ {row['grokking_step_std']:.0f}" if not pd.isna(row.get('grokking_step_std')) else f"{row['grokking_step_mean']:.0f}"

        row_str.extend([test_acc, weight_norm, fourier, grokked_frac, grok_step])
        latex_str += " & ".join(row_str) + " \\\\\n"

    latex_str += "\\bottomrule\n\\end{tabular}\n\\caption{Aggregated experiment results.}\n\\label{tab:aggregated_results}\n\\end{table*}"

    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"LaTeX table saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate Results")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing results")
    parser.add_argument("--output_file", type=str, default="results/aggregated_table.tex", help="Output LaTeX file")

    args = parser.parse_args()

    results_files = find_result_files(args.results_dir)
    print(f"Found {len(results_files)} results files.")

    df = aggregate_metrics(results_files)
    if df.empty:
        print("No valid data found to aggregate.")
        return

    group_by_cols = ["collapse_ratio", "noise_fraction", "weight_decay"]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    generate_latex_table(df, group_by_cols, args.output_file)

    # Also save the raw aggregated data
    csv_out = str(Path(args.output_file).with_suffix('.csv'))
    df.to_csv(csv_out, index=False)
    print(f"Raw aggregated data saved to {csv_out}")

if __name__ == "__main__":
    main()
