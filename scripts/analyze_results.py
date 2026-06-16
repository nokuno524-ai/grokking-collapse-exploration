#!/usr/bin/env python3
"""
Comprehensive results analysis script.
Loads all results from results/ directory across conditions and seeds.
Computes summary statistics, performs Mann-Whitney U tests, exports tables,
and generates summary figures.
"""

import json
import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def gather_results(results_dir="results"):
    data = []

    # We expect results to be in results/<condition>/seed_*/results.json
    # or results/<condition>/results.json
    # Let's search recursively
    pattern = os.path.join(results_dir, "**", "results.json")
    for filepath in glob.glob(pattern, recursive=True):
        try:
            with open(filepath, 'r') as f:
                res = json.load(f)

            # Try to infer condition and seed
            parts = Path(filepath).parts
            # Usually results/condition/seed_X/results.json
            if len(parts) >= 4 and parts[-2].startswith("seed_"):
                condition = parts[-3]
                seed = parts[-2]
            elif len(parts) >= 3:
                condition = parts[-2]
                seed = "seed_unknown"
            else:
                condition = "unknown"
                seed = "unknown"

            cfg = res.get("config", {})
            grokked = res.get("grokked", False)
            grokking_step = res.get("grokking_step", None)
            final_test_acc = res.get("final_test_acc", 0.0)
            final_train_acc = res.get("final_train_acc", 0.0)

            # Compute loss trajectory features from history
            history = res.get("history", [])
            plateau_level = 0.0
            transition_sharpness = 0.0

            if history:
                test_accs = [h.get("test_acc", 0) for h in history]
                # Plateau level: avg of last 10% of training
                n_tail = max(1, len(test_accs) // 10)
                plateau_level = np.mean(test_accs[-n_tail:])

                if grokked and grokking_step is not None:
                    # Sharpness: max derivative of test acc around grokking
                    diffs = np.diff(test_accs)
                    transition_sharpness = np.max(diffs) if len(diffs) > 0 else 0.0

            data.append({
                "condition": condition,
                "seed": seed,
                "grokked": grokked,
                "grokking_step": grokking_step,
                "final_test_acc": final_test_acc,
                "final_train_acc": final_train_acc,
                "plateau_level": plateau_level,
                "transition_sharpness": transition_sharpness,
                "filepath": filepath,
                "history": history
            })
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")

    return pd.DataFrame(data)


def compute_statistics(df, output_dir="analysis"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Summary stats
    summary = df.groupby("condition").agg({
        "grokked": "mean",
        "grokking_step": ["mean", "std"],
        "final_test_acc": ["mean", "std"],
        "transition_sharpness": "mean"
    }).round(4)

    # Flatten multi-level columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.reset_index(inplace=True)

    csv_path = os.path.join(output_dir, "results_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")

    # Export LaTeX table
    tex_path = os.path.join(output_dir, "results_summary.tex")
    summary.to_latex(tex_path, index=False)
    print(f"Saved summary LaTeX to {tex_path}")

    # 2. Mann-Whitney U test (Pure vs others for grokking rates)
    # The pure condition is our baseline.
    pure_grokked = df[df["condition"].str.contains("pure", case=False, na=False)]["grokked"].astype(float).values

    stats_data = []
    if len(pure_grokked) > 0:
        for cond in df["condition"].unique():
            if "pure" not in cond.lower():
                cond_grokked = df[df["condition"] == cond]["grokked"].astype(float).values
                if len(cond_grokked) > 0:
                    try:
                        u_stat, p_val = stats.mannwhitneyu(pure_grokked, cond_grokked, alternative="two-sided")
                        stats_data.append({
                            "condition": cond,
                            "u_statistic": u_stat,
                            "p_value": p_val
                        })
                    except Exception as e:
                        pass

    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_csv = os.path.join(output_dir, "mann_whitney_tests.csv")
        stats_df.to_csv(stats_csv, index=False)
        print(f"Saved Mann-Whitney test results to {stats_csv}")

    return summary


def plot_summary_figures(df, output_dir="analysis"):
    os.makedirs(output_dir, exist_ok=True)

    # Define a consistent color palette
    conditions = sorted(df["condition"].unique())
    palette = sns.color_palette("husl", len(conditions))
    color_map = dict(zip(conditions, palette))

    # Filter conditions for cleaner plots if needed, or use all

    # 1. Accuracy curves overlaid by condition
    plt.figure(figsize=(10, 6))

    for idx, row in df.iterrows():
        history = row["history"]
        if history:
            steps = [h.get("step") for h in history]
            test_accs = [h.get("test_acc") for h in history]
            cond = row["condition"]

            # Only label the first line of each condition for the legend
            label = cond if cond not in plt.gca().get_legend_handles_labels()[1] else ""
            plt.plot(steps, test_accs, color=color_map[cond], alpha=0.3, label=label)

    # Need to deduplicate legend handles
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())

    plt.title("Test Accuracy Trajectories by Condition")
    plt.xlabel("Step")
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "accuracy_trajectories.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Gradient noise scale evolution
    plt.figure(figsize=(10, 6))
    for idx, row in df.iterrows():
        history = row["history"]
        if history:
            steps = [h.get("step") for h in history]
            # Assumes gradient_noise_scale might be present in history
            gns = [h.get("gradient_noise_scale") for h in history if "gradient_noise_scale" in h]
            if not gns:
                continue
            cond = row["condition"]
            label = cond if cond not in plt.gca().get_legend_handles_labels()[1] else ""
            if len(steps) == len(gns):
                plt.plot(steps, gns, color=color_map[cond], alpha=0.3, label=label)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())

    plt.title("Gradient Noise Scale Evolution")
    plt.xlabel("Step")
    plt.ylabel("Noise Scale")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "gradient_noise_scale.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Weight norm evolution
    plt.figure(figsize=(10, 6))
    for idx, row in df.iterrows():
        history = row["history"]
        if history:
            steps = [h.get("step") for h in history]
            norms = [h.get("weight_norm") for h in history if "weight_norm" in h]
            if not norms:
                continue
            cond = row["condition"]

            label = cond if cond not in plt.gca().get_legend_handles_labels()[1] else ""
            # Align steps and norms if some are missing
            if len(steps) == len(norms):
                plt.plot(steps, norms, color=color_map[cond], alpha=0.3, label=label)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys())

    plt.title("Weight Norm Evolution")
    plt.xlabel("Step")
    plt.ylabel("L2 Norm")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "weight_norm_evolution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved summary plots.")

if __name__ == "__main__":
    df = gather_results()
    if len(df) == 0:
        print("No results found in results/ directory.")
    else:
        print(f"Loaded {len(df)} result files.")
        summary = compute_statistics(df)
        plot_summary_figures(df)
