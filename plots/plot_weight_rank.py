import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'figure.dpi': 300})

COLORS = {
    "pure": "#0072B2",
    "medium_collapse": "#009E73",
    "severe_collapse": "#D55E00"
}

def plot_weight_rank(data_dir: Path, output_file: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    conditions = ["pure", "medium_collapse", "severe_collapse"]

    for c in conditions:
        csv_path = data_dir / f"{c}_forensics.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        steps = df['step']

        ax1.plot(steps, df['embed_rank'], label=c, color=COLORS[c], linewidth=2)
        ax2.plot(steps, df['dist_from_init'], label=c, color=COLORS[c], linewidth=2)

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Effective Rank")
    ax1.set_title("Token Embedding Rank Evolution")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("L2 Distance")
    ax2.set_title("Distance from Initialization")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved {output_file}")

if __name__ == "__main__":
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    plot_weight_rank(Path("analysis/weight_forensics"), out_dir / "weight_rank_evolution.pdf")
