import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'figure.dpi': 300})

def plot_head_heatmap(data_dir: Path, output_file: Path):
    conditions = ["pure", "medium_collapse", "severe_collapse"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i, c in enumerate(conditions):
        csv_path = data_dir / f"{c}_gates.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        head_cols = [col for col in df.columns if col.startswith("head_")]

        heatmap_data = df[head_cols].T

        sns.heatmap(heatmap_data, ax=axes[i], cmap="viridis", cbar=(i==2),
                    xticklabels=max(1, len(df)//5))

        axes[i].set_title(c.replace("_", " ").title())
        axes[i].set_xlabel("Checkpoint Index")
        if i == 0:
            axes[i].set_ylabel("Attention Head")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved {output_file}")

if __name__ == "__main__":
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    plot_head_heatmap(Path("analysis/causal_head_gating"), out_dir / "head_roles_heatmap.pdf")
