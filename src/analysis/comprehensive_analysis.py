import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_results_dir(base_dir):
    data = []
    # Collapse levels from get_all_conditions
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    for cond in conditions:
        cond_dir = os.path.join(base_dir, cond)
        res_file = os.path.join(cond_dir, "results.json")
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                res = json.load(f)

            history = res.get("history", [])
            for pt in history:
                data.append({
                    "condition": cond,
                    "collapse_level": res["config"].get("collapse_level", 0.0),
                    "grokked": res.get("grokked", False),
                    "grokking_step": res.get("grokking_step", None),
                    "step": pt["step"],
                    "train_loss": pt["train_loss"],
                    "test_loss": pt["test_loss"],
                    "train_acc": pt["train_acc"],
                    "test_acc": pt["test_acc"],
                    "weight_norm": pt["weight_norm"]
                })
    return pd.DataFrame(data)

def plot_analysis(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Loss curves
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="step", y="test_loss", hue="condition")
    plt.title("Test Loss vs Step across Collapse Levels")
    plt.yscale('log')
    plt.savefig(os.path.join(out_dir, "test_loss_curves.png"))
    plt.savefig(os.path.join(out_dir, "test_loss_curves.pdf"))
    plt.close()

    # 2. Test accuracy curves
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="step", y="test_acc", hue="condition")
    plt.title("Test Accuracy vs Step across Collapse Levels")
    plt.savefig(os.path.join(out_dir, "test_accuracy_curves.png"))
    plt.savefig(os.path.join(out_dir, "test_accuracy_curves.pdf"))
    plt.close()

    # 3. Weight norm evolution
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="step", y="weight_norm", hue="condition")
    plt.title("Weight Norm vs Step across Collapse Levels")
    plt.savefig(os.path.join(out_dir, "weight_norm_evolution.png"))
    plt.savefig(os.path.join(out_dir, "weight_norm_evolution.pdf"))
    plt.close()

def compute_statistics(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stats = []

    # Get one row per condition
    summary = df.groupby("condition").first().reset_index()
    for _, row in summary.iterrows():
        stats.append({
            "condition": row["condition"],
            "collapse_level": row["collapse_level"],
            "grokked": row["grokked"],
            "grokking_step": row["grokking_step"]
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(out_dir, "grokking_statistics.csv"), index=False)
    print("Saved grokking statistics.")
    print(stats_df)

if __name__ == "__main__":
    df = parse_results_dir("results")
    if not df.empty:
        plot_analysis(df, "analysis/comprehensive")
        compute_statistics(df, "analysis/comprehensive")
    else:
        print("No results found to analyze.")
