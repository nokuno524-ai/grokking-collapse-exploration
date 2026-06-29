import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300
    })

def load_data(results_dir="results/phase_transitions"):
    p = Path(results_dir)
    data = []
    if not p.exists():
        return pd.DataFrame()

    for cond_dir in p.iterdir():
        if not cond_dir.is_dir(): continue
        for seed_dir in cond_dir.iterdir():
            if not seed_dir.is_dir(): continue
            res_file = seed_dir / "results.json"
            if res_file.exists():
                try:
                    with open(res_file, 'r') as f:
                        res = json.load(f)
                    cfg = res['config']
                    data.append({
                        'noise': cfg.get('noise_fraction', 0.0),
                        'collapse': cfg.get('collapse_level', 0.0),
                        'wd': cfg.get('weight_decay', 1.0),
                        'test_acc': res.get('final_test_acc', 0.0),
                        'fourier': res.get('final_fourier_concentration', 0.0),
                        'grokked': 1.0 if res.get('grokked', False) else 0.0
                    })
                except Exception:
                    pass
    return pd.DataFrame(data)

def generate_main_grid(df: pd.DataFrame, out_dir="visualizations"):
    if df.empty:
        return

    set_style()
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    # 2x3 grid: Top row=Grokking Rate, Bottom row=Test Acc
    # Cols: wd=0.0, wd=1.0, wd=3.0

    wds = [0.0, 1.0, 3.0]

    for i, wd in enumerate(wds):
        sub_df = df[df['wd'] == wd]
        if sub_df.empty:
            continue

        # Pivot for grokking rate
        agg_grok = sub_df.groupby(['noise', 'collapse'])['grokked'].mean().reset_index()
        pivot_grok = agg_grok.pivot(index='noise', columns='collapse', values='grokked')

        # Pivot for test acc
        agg_acc = sub_df.groupby(['noise', 'collapse'])['test_acc'].mean().reset_index()
        pivot_acc = agg_acc.pivot(index='noise', columns='collapse', values='test_acc')

        sns.heatmap(pivot_grok, ax=axes[0, i], cmap="viridis", vmin=0, vmax=1,
                    cbar_kws={'label': 'Grokking Prob'})
        axes[0, i].set_title(f"Grokking Rate (wd={wd})")
        axes[0, i].invert_yaxis()

        sns.heatmap(pivot_acc, ax=axes[1, i], cmap="plasma", vmin=0, vmax=1,
                    cbar_kws={'label': 'Test Accuracy'})
        axes[1, i].set_title(f"Final Test Acc (wd={wd})")
        axes[1, i].invert_yaxis()

    plt.savefig(Path(out_dir) / "main_result_grid.png")
    plt.close()

def generate_mechanistic_comparison(df: pd.DataFrame, out_dir="visualizations"):
    if df.empty:
        return
    set_style()
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.scatterplot(data=df, x='test_acc', y='fourier', hue='grokked',
                    palette={0: "red", 1: "green"}, alpha=0.7, ax=ax)

    ax.set_title("Test Accuracy vs Fourier Concentration")
    ax.set_xlabel("Test Accuracy")
    ax.set_ylabel("Fourier Concentration")

    plt.savefig(Path(out_dir) / "mechanistic_comparison.png")
    plt.close()

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        generate_main_grid(df)
        generate_mechanistic_comparison(df)
        print("Generated publication figures in visualizations/")
    else:
        print("No data found to plot.")
