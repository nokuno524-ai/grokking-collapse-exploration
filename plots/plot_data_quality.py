import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'figure.dpi': 300})

def plot_data_quality(data_file: Path, results_dir: Path, output_file: Path):
    if not data_file.exists():
        return

    df = pd.read_csv(data_file)

    # Extract grokking success metrics
    grokking_success = []
    for c in df['condition']:
        cond_dir = results_dir / c
        if cond_dir.exists() and (cond_dir / "results.json").exists():
            with open(cond_dir / "results.json") as f:
                res = json.load(f)
            grokking_success.append(res.get('final_test_acc', 0.0))
        else:
            grokking_success.append(0.0)

    df['grokking_success'] = grokking_success

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df['collapse_level'], df['kl_divergence'], 'o-', linewidth=2, color="#0072B2")
    axes[0].set_xlabel("Collapse Level (Fraction)")
    axes[0].set_ylabel("KL Divergence (nats)")
    axes[0].set_title("Distribution Shift vs Collapse")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['collapse_level'], df['entropy'], 'o-', linewidth=2, color="#D55E00")
    axes[1].set_xlabel("Collapse Level (Fraction)")
    axes[1].set_ylabel("Shannon Entropy")
    axes[1].set_title("Target Diversity vs Collapse")
    axes[1].grid(True, alpha=0.3)

    # Correlation plot
    scatter = axes[2].scatter(df['kl_divergence'], df['grokking_success'], c=df['collapse_level'], cmap='viridis', s=100)
    axes[2].plot(df['kl_divergence'], df['grokking_success'], 'k--', alpha=0.3)
    axes[2].set_xlabel("KL Divergence (nats)")
    axes[2].set_ylabel("Final Test Accuracy")
    axes[2].set_title("Grokking Success vs Data Quality")
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2], label='Collapse Level')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved {output_file}")

if __name__ == "__main__":
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    plot_data_quality(Path("analysis/data_metrics/dataset_metrics.csv"), Path("results"), out_dir / "data_quality.pdf")
