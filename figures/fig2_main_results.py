import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
import pandas as pd

def parse_results_dir(base_dir, condition_name):
    """Parse results.json files for a given condition across seeds."""
    condition_path = os.path.join(base_dir, condition_name)
    seed_dirs = glob(os.path.join(condition_path, "seed_*"))

    all_history = []

    for seed_dir in seed_dirs:
        json_path = os.path.join(seed_dir, "results.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            history = data.get('history', [])
            for h in history:
                all_history.append({
                    'seed': os.path.basename(seed_dir),
                    'step': h['step'],
                    'train_acc': h['train_acc'],
                    'test_acc': h['test_acc']
                })

    if not all_history:
        # Try finding json in the root if no seed dirs
        json_path = os.path.join(condition_path, "results.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            history = data.get('history', [])
            for h in history:
                all_history.append({
                    'seed': 'seed_0',
                    'step': h['step'],
                    'train_acc': h['train_acc'],
                    'test_acc': h['test_acc']
                })

    return pd.DataFrame(all_history)

def plot_main_results(results_base_dir, output_path):
    """
    Plot test/train accuracy curves with confidence intervals.
    """
    sns.set_theme(style="whitegrid", context="paper")

    # We will plot Pure vs. Low Collapse vs. Severe Collapse
    conditions = {
        'pure': 'Pure Data (0%)',
        'low_collapse': 'Low Collapse (5%)',
        'severe_collapse': 'Severe Collapse (30%)'
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = sns.color_palette("Set1", n_colors=len(conditions))

    for (cond_id, cond_label), color in zip(conditions.items(), colors):
        df = parse_results_dir(results_base_dir, cond_id)
        if df.empty:
            print(f"Warning: No data found for {cond_id}")
            continue

        # Group by step to get mean and std
        # Drop non-numeric columns like 'seed' before aggregation
        grouped = df.drop(columns=['seed']).groupby('step')
        steps = grouped.mean().index

        train_mean = grouped['train_acc'].mean().values
        train_std = grouped['train_acc'].std().fillna(0).values

        test_mean = grouped['test_acc'].mean().values
        test_std = grouped['test_acc'].std().fillna(0).values

        # Plot Train
        axes[0].plot(steps, train_mean, label=cond_label, color=color)
        axes[0].fill_between(steps, train_mean - train_std, train_mean + train_std, alpha=0.2, color=color)

        # Plot Test
        axes[1].plot(steps, test_mean, label=cond_label, color=color)
        axes[1].fill_between(steps, test_mean - test_std, test_mean + test_std, alpha=0.2, color=color)

    axes[0].set_title('Training Accuracy')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    axes[1].set_title('Test Accuracy (Grokking)')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    # Add a horizontal line for grokking threshold
    axes[1].axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='Grokking Threshold')

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300)
    plt.savefig(f"{output_path}.pdf")
    plt.close()

if __name__ == "__main__":
    plot_main_results("results", "figures/fig2_main_results")
    print("Generated figures/fig2_main_results.png/pdf")
