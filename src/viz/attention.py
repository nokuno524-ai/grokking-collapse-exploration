import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional
import seaborn as sns

def plot_attention_heatmaps(
    attention_weights: np.ndarray,
    layer_idx: int,
    output_path: str,
    title_suffix: str = "",
    input_tokens: Optional[List[int]] = None
) -> None:
    """
    Plots attention heatmaps for all heads in a given layer for a specific input example.

    Args:
        attention_weights: Array of shape (n_heads, seq_len, seq_len)
        layer_idx: Layer index being plotted
        output_path: Path to save the figure
        title_suffix: Optional suffix for the main title
        input_tokens: Optional list of tokens (e.g., [a, b]) for axis labels
    """
    n_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    labels = [str(t) for t in input_tokens] if input_tokens else [f"Pos {i}" for i in range(attention_weights.shape[-1])]

    for h in range(n_heads):
        sns.heatmap(
            attention_weights[h],
            ax=axes[h],
            cmap="viridis",
            vmin=0, vmax=1,
            annot=True, fmt=".2f",
            xticklabels=labels, yticklabels=labels
        )
        axes[h].set_title(f"Head {h}")
        axes[h].set_xlabel("Key")
        axes[h].set_ylabel("Query")

    plt.suptitle(f"Layer {layer_idx} Attention{title_suffix}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_entropy_trajectories(
    steps: List[int],
    entropies_per_condition: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    layer_idx: int = 0
) -> None:
    """
    Plots attention entropy vs training step across different conditions.

    Args:
        steps: List of training steps
        entropies_per_condition: Dict mapping condition_name -> { "head_x": entropy_array }
                                 where entropy_array has same length as steps.
        output_path: Path to save the figure
        layer_idx: Layer index for title context
    """
    if not steps:
        return

    # We will create a plot for each head
    sample_cond = list(entropies_per_condition.keys())[0]
    heads = list(entropies_per_condition[sample_cond].keys())
    n_heads = len(heads)

    fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for i, head_name in enumerate(heads):
        ax = axes[i]
        for cond, metrics in entropies_per_condition.items():
            if head_name in metrics:
                ax.plot(steps, metrics[head_name], marker='o', label=cond, markersize=3)
        ax.set_title(f"{head_name.capitalize()}")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Attention Entropy")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    plt.suptitle(f"Layer {layer_idx} Attention Entropy vs Training Step")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_head_clustering(
    similarity_matrix: np.ndarray,
    head_labels: List[str],
    output_path: str,
    title: str = "Attention Head Similarity"
) -> None:
    """
    Plots a clustered heatmap of attention heads based on a similarity matrix.

    Args:
        similarity_matrix: Square matrix of shape (n_total_heads, n_total_heads)
        head_labels: Labels for each head (e.g., "L0H0", "L0H1")
        output_path: Path to save the figure
        title: Title of the plot
    """
    plt.figure(figsize=(8, 6))

    # We use seaborn clustermap for automatic hierarchical clustering
    g = sns.clustermap(
        similarity_matrix,
        xticklabels=head_labels,
        yticklabels=head_labels,
        cmap="coolwarm",
        annot=False,
        figsize=(8, 8)
    )

    g.fig.suptitle(title, y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(g.fig)
