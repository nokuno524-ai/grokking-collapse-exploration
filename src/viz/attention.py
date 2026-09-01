import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def plot_attention_entropy_over_time(
    steps: List[int],
    entropies: List[float],
    title: str,
    output_path: Path,
    csv_path: Path
):
    """
    Plots mean attention entropy over time and saves to PNG and CSV.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(steps, entropies, marker='o', linestyle='-')
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Attention Entropy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    # Save CSV
    df = pd.DataFrame({'step': steps, 'mean_entropy': entropies})
    df.to_csv(csv_path, index=False)


def plot_head_specialization_heatmap(
    cluster_labels: np.ndarray,
    n_layers: int,
    n_heads: int,
    title: str,
    output_path: Path,
    csv_path: Path
):
    """
    Plots a heatmap of head cluster assignments (n_layers x n_heads).
    """
    # Reshape to (n_layers, n_heads)
    grid = cluster_labels.reshape((n_layers, n_heads))

    plt.figure(figsize=(max(4, n_heads), max(3, n_layers)))
    # Use a categorical colormap
    cmap = plt.get_cmap('tab10', len(np.unique(grid)))

    im = plt.imshow(grid, cmap=cmap, aspect='auto')
    plt.colorbar(im, ticks=np.unique(grid), label='Cluster ID')

    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.title(title)

    # Add text annotations
    for i in range(n_layers):
        for j in range(n_heads):
            plt.text(j, i, str(grid[i, j]), ha='center', va='center', color='white' if cmap(grid[i, j])[0] < 0.5 else 'black')

    plt.xticks(np.arange(n_heads))
    plt.yticks(np.arange(n_layers))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    # Save CSV
    rows = []
    for i in range(n_layers):
        for j in range(n_heads):
            rows.append({'layer': i, 'head': j, 'cluster': grid[i, j]})
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


def plot_diagnostic_token_traces(
    steps: List[int],
    traces: Dict[str, List[float]],
    title: str,
    output_path: Path,
    csv_path: Path
):
    """
    Plots attention trace curves (e.g. attention from pos 1 to pos 0) over time.
    """
    plt.figure(figsize=(8, 5))
    for label, vals in traces.items():
        plt.plot(steps, vals, marker='.', label=label)

    plt.xlabel('Training Steps')
    plt.ylabel('Attention Weight')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    # Save CSV
    df = pd.DataFrame({'step': steps})
    for label, vals in traces.items():
        df[label] = vals
    df.to_csv(csv_path, index=False)
