import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from glob import glob

def parse_mechanism_data(results_base_dir, condition_name):
    """Parse results.json files for mechanism data (weight norm, rank, fourier)."""
    condition_path = os.path.join(results_base_dir, condition_name)
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
                    'weight_norm': h.get('weight_norm', np.nan),
                    'embedding_rank': h.get('embedding_rank', np.nan),
                    'fourier_concentration': h.get('fourier_concentration', np.nan)
                })

    if not all_history:
        json_path = os.path.join(condition_path, "results.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            history = data.get('history', [])
            for h in history:
                all_history.append({
                    'seed': 'seed_0',
                    'step': h['step'],
                    'weight_norm': h.get('weight_norm', np.nan),
                    'embedding_rank': h.get('embedding_rank', np.nan),
                    'fourier_concentration': h.get('fourier_concentration', np.nan)
                })

    return pd.DataFrame(all_history)

def plot_mechanism_figures(results_base_dir, output_path):
    """
    Plot weight norms, attention entropy, and Fourier spectrum side-by-side.
    """
    sns.set_theme(style="whitegrid", context="paper")

    conditions = {
        'pure': 'Pure Data (0%)',
        'severe_collapse': 'Severe Collapse (30%)'
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = sns.color_palette("Set1", n_colors=len(conditions))

    for (cond_id, cond_label), color in zip(conditions.items(), colors):
        df = parse_mechanism_data(results_base_dir, cond_id)
        if df.empty:
            continue

        grouped = df.drop(columns=['seed']).groupby('step')
        steps = grouped.mean().index

        # 1. Weight Norm
        wn_mean = grouped['weight_norm'].mean().values
        wn_std = grouped['weight_norm'].std().fillna(0).values
        axes[0].plot(steps, wn_mean, label=cond_label, color=color)
        axes[0].fill_between(steps, wn_mean - wn_std, wn_mean + wn_std, alpha=0.2, color=color)

        # 2. Embedding Rank
        rank_mean = grouped['embedding_rank'].mean().values
        rank_std = grouped['embedding_rank'].std().fillna(0).values
        axes[1].plot(steps, rank_mean, label=cond_label, color=color)
        axes[1].fill_between(steps, rank_mean - rank_std, rank_mean + rank_std, alpha=0.2, color=color)

        # 3. Fourier Concentration
        fc_mean = grouped['fourier_concentration'].mean().values
        fc_std = grouped['fourier_concentration'].std().fillna(0).values
        axes[2].plot(steps, fc_mean, label=cond_label, color=color)
        axes[2].fill_between(steps, fc_mean - fc_std, fc_mean + fc_std, alpha=0.2, color=color)

    axes[0].set_title('Total Weight Norm')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('L2 Norm')
    axes[0].legend()

    # Note: Using Embedding Rank as a proxy for structural entropy during grokking
    # since per-head attention entropy requires loading all checkpoints which is slow.
    axes[1].set_title('Embedding Rank (Structural Entropy)')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Effective Rank')

    axes[2].set_title('Fourier Concentration')
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('Concentration')

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300)
    plt.savefig(f"{output_path}.pdf")
    plt.close()

if __name__ == "__main__":
    plot_mechanism_figures("results", "figures/fig3_mechanism")
    print("Generated figures/fig3_mechanism.png/pdf")
