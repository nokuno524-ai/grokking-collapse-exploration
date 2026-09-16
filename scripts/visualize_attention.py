import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to python path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_attention_heatmaps(npz_path: Path, out_path: Path):
    """
    Plots attention heatmaps for each head in a single batch sample.
    Assumes layer 0 if multiple layers exist.
    """
    data = np.load(npz_path)
    key = 'layer_0' if 'layer_0' in data else 'attention'
    attn_matrix = data[key]
    n_heads = attn_matrix.shape[1]
    sample = attn_matrix[0] # Take first sample

    fig, axes = plt.subplots(1, n_heads, figsize=(3*n_heads, 3))
    if n_heads == 1:
        axes = [axes]

    for i in range(n_heads):
        im = axes[i].imshow(sample[i], cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(f'Head {i}')
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xlabel('Key')
        axes[i].set_ylabel('Query')

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_entropy_timeline(csv_path: Path, out_path: Path, layer_idx: int = 0):
    """
    Plots head-activity timelines across training (entropy of attention distributions).
    """
    df = pd.read_csv(csv_path)
    if 'layer' in df.columns:
        df = df[df['layer'] == layer_idx]

    steps = df['step']

    # Find all entropy columns
    entropy_cols = [c for c in df.columns if c.startswith('head_') and c.endswith('_entropy')]

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in entropy_cols:
        head_num = col.split('_')[1]
        ax.plot(steps, df[col], label=f'Head {head_num}')

    ax.plot(steps, df['mean_entropy'], label='Mean Entropy', linestyle='--', color='black', linewidth=2)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Attention Entropy')
    ax.set_title(f'Attention Entropy over Training (Layer {layer_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_overlay_comparison(pure_csv: Path, contam_csv: Path, out_path: Path, metric: str = 'mean_entropy', layer_idx: int = 0):
    """
    Overlay plots comparing pure vs. collapse runs at matched steps.
    """
    df_pure = pd.read_csv(pure_csv)
    df_contam = pd.read_csv(contam_csv)

    if 'layer' in df_pure.columns:
        df_pure = df_pure[df_pure['layer'] == layer_idx]
    if 'layer' in df_contam.columns:
        df_contam = df_contam[df_contam['layer'] == layer_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_pure['step'], df_pure[metric], label=f'Pure ({metric})', color='blue')
    ax.plot(df_contam['step'], df_contam[metric], label=f'Collapse ({metric})', color='red', linestyle='--')

    ax.set_xlabel('Training Step')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison (Layer {layer_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize extracted attention and metrics.")
    parser.add_argument("--heatmap-npz", type=Path, help="Path to an NPZ file to plot heatmaps from")
    parser.add_argument("--timeline-csv", type=Path, help="Path to a CSV file to plot entropy timeline")
    parser.add_argument("--overlay-pure", type=Path, help="Path to pure CSV for overlay")
    parser.add_argument("--overlay-contam", type=Path, help="Path to contam CSV for overlay")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory for plots")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.heatmap_npz:
        out = args.out_dir / f"heatmaps_{args.heatmap_npz.stem}.png"
        plot_attention_heatmaps(args.heatmap_npz, out)
        print(f"Saved {out}")

    if args.timeline_csv:
        out = args.out_dir / f"timeline_{args.timeline_csv.stem}.png"
        plot_entropy_timeline(args.timeline_csv, out)
        print(f"Saved {out}")

    if args.overlay_pure and args.overlay_contam:
        out = args.out_dir / "overlay_comparison.png"
        plot_overlay_comparison(args.overlay_pure, args.overlay_contam, out)
        print(f"Saved {out}")

if __name__ == "__main__":
    main()
