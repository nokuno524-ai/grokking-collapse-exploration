import os
import json
import pandas as pd
from pathlib import Path

def load_results(output_dir: str) -> pd.DataFrame:
    """
    Traverse the output directory, load all results.json files,
    and aggregate them into a pandas DataFrame.
    """
    root_path = Path(output_dir)
    all_data = []

    for results_file in root_path.rglob("results.json"):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)

            config = data.get("config", {})

            # Extract final metrics
            entry = {
                "condition_name": config.get("condition_name"),
                "seed": config.get("seed"),
                "collapse_level": config.get("collapse_level"),
                "collapse_severity": config.get("collapse_severity"),
                "grokked": data.get("grokked"),
                "grokking_step": data.get("grokking_step"),
                "final_train_acc": data.get("final_train_acc"),
                "final_test_acc": data.get("final_test_acc"),
                "final_weight_norm": data.get("final_weight_norm"),
                "final_embedding_rank": data.get("final_embedding_rank"),
                "final_fourier_concentration": data.get("final_fourier_concentration"),
                "history": data.get("history", [])
            }
            all_data.append(entry)
        except Exception as e:
            print(f"Failed to process {results_file}: {e}")

    df = pd.DataFrame(all_data)
    return df

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(df: pd.DataFrame, output_dir: str):
    """
    Generate plots for training curves (accuracy/loss over time),
    weight norm, and fourier concentration.
    """
    if df.empty or "history" not in df.columns:
        print("Empty DataFrame or no history, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Expand history into a long-form DataFrame for seaborn
    records = []
    for _, row in df.iterrows():
        hist = row["history"]
        cond = f"ratio_{row['collapse_level']}_sev_{row['collapse_severity']}"
        seed = row["seed"]

        for entry in hist:
            records.append({
                "condition": cond,
                "collapse_level": row["collapse_level"],
                "seed": seed,
                "step": entry.get("step"),
                "train_acc": entry.get("train_acc"),
                "test_acc": entry.get("test_acc"),
                "train_loss": entry.get("train_loss"),
                "test_loss": entry.get("test_loss"),
                "weight_norm": entry.get("weight_norm")
            })

    if not records:
        return

    hist_df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid")

    # Plot Test Accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hist_df, x="step", y="test_acc", hue="collapse_level", palette="viridis", errorbar="sd")
    plt.title("Test Accuracy over Time")
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.axhline(0.95, ls="--", color="gray", label="Grokking Threshold")
    plt.savefig(os.path.join(output_dir, "test_accuracy.png"), dpi=300)
    plt.close()

    # Plot Weight Norm
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hist_df, x="step", y="weight_norm", hue="collapse_level", palette="viridis", errorbar="sd")
    plt.title("Weight Norm over Time")
    plt.ylabel("L2 Norm")
    plt.xlabel("Step")
    plt.savefig(os.path.join(output_dir, "weight_norm.png"), dpi=300)
    plt.close()

def generate_summary_table(df: pd.DataFrame, output_csv: str) -> pd.DataFrame:
    """
    Generate a summary table computing mean and std over seeds.
    """
    if df.empty:
        print("Empty DataFrame, skipping summary table.")
        return df

    # Group by collapse_level and collapse_severity
    groupby_cols = ["collapse_level", "collapse_severity"]
    metrics = [
        "final_train_acc", "final_test_acc", "final_weight_norm",
        "final_embedding_rank", "final_fourier_concentration"
    ]

    # We will compute mean and std
    summary = df.groupby(groupby_cols)[metrics].agg(['mean', 'std']).reset_index()

    # Flatten columns
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    summary.to_csv(output_csv, index=False)
    print(f"Summary table saved to {output_csv}")
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results", help="Directory containing experiment results")
    parser.add_argument("--summary-file", type=str, default="summary.csv", help="Path to save the summary CSV")
    args = parser.parse_args()

    df = load_results(args.output_dir)
    if not df.empty:
        print(f"Loaded {len(df)} results.")
        generate_summary_table(df, os.path.join(args.output_dir, args.summary_file) if not os.path.isabs(args.summary_file) else args.summary_file)
        generate_plots(df, args.output_dir)
        print(f"Plots generated in {args.output_dir}")
    else:
        print("No results found.")
