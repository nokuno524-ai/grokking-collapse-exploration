"""
Visualization functions for attention maps.
Generates heatmaps, difference heatmaps, and saves them to PNGs.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, List, Tuple


def plot_attention_heatmap(
    attn: Union[torch.Tensor, np.ndarray],
    layer_idx: int,
    head_idx: int,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar: bool = True,
):
    """
    Plot a single attention heatmap for a specific layer and head.

    Args:
        attn: Attention weights of shape (seq_len, seq_len)
        layer_idx: Layer index for title
        head_idx: Head index for title
        ax: Optional matplotlib axes to plot on
        title: Optional custom title string
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cbar: Whether to include a colorbar
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    if isinstance(attn, torch.Tensor):
        attn = attn.cpu().numpy()

    sns.heatmap(
        attn,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=cbar,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={'shrink': 0.8}
    )

    if title is None:
        title = f"Layer {layer_idx}, Head {head_idx}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Key Position", fontsize=9)
    ax.set_ylabel("Query Position", fontsize=9)

    # Format ticks to be minimal if sequences are small
    seq_len = attn.shape[0]
    if seq_len <= 5:
        ax.set_xticks(np.arange(seq_len) + 0.5)
        ax.set_yticks(np.arange(seq_len) + 0.5)
        ax.set_xticklabels([f"{i}" for i in range(seq_len)])
        ax.set_yticklabels([f"{i}" for i in range(seq_len)], rotation=0)


def plot_attention_grid(
    attn: Union[torch.Tensor, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Attention Patterns",
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    Plot a grid of attention heatmaps for all layers and heads.

    Args:
        attn: Attention weights of shape (n_layers, n_heads, seq_len, seq_len)
        output_path: Where to save the plot (if None, will only create figure)
        title: Title for the entire figure
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        Matplotlib Figure object
    """
    if isinstance(attn, torch.Tensor):
        attn = attn.cpu().numpy()

    n_layers, n_heads, _, _ = attn.shape

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(4 * n_heads, 3.5 * n_layers),
        squeeze=False
    )

    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            plot_attention_heatmap(
                attn[l, h],
                layer_idx=l,
                head_idx=h,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar=(h == n_heads - 1)  # Only add colorbar to the last head in each row
            )

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_attention_diff_grid(
    attn_a: Union[torch.Tensor, np.ndarray],
    attn_b: Union[torch.Tensor, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Attention Difference (A - B)",
    cmap: str = "RdBu_r",
    vmax_diff: float = 0.5,
):
    """
    Plot a grid of attention difference heatmaps for all layers and heads.
    Shows attn_a - attn_b.

    Args:
        attn_a: Attention weights A of shape (n_layers, n_heads, seq_len, seq_len)
        attn_b: Attention weights B of shape (n_layers, n_heads, seq_len, seq_len)
        output_path: Where to save the plot
        title: Title for the entire figure
        cmap: Diverging colormap name (default: RdBu_r, where red is positive, blue is negative)
        vmax_diff: Maximum absolute difference for colormap scaling

    Returns:
        Matplotlib Figure object
    """
    if isinstance(attn_a, torch.Tensor):
        attn_a = attn_a.cpu().numpy()
    if isinstance(attn_b, torch.Tensor):
        attn_b = attn_b.cpu().numpy()

    diff = attn_a - attn_b

    return plot_attention_grid(
        diff,
        output_path=output_path,
        title=title,
        cmap=cmap,
        vmin=-vmax_diff,
        vmax=vmax_diff
    )
