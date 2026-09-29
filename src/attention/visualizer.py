"""
Visualization suite for transformer attention patterns.

Provides reusable plotting functions for:
- Per-layer/per-head attention heatmap grids
- Attention entropy over training steps
- Head-specialization clustering plots (hierarchical clustering)
- Animating step-by-step attention pattern evolution
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def plot_attention_grid(
    attention_tensor: np.ndarray,
    output_path: Union[str, Path],
    layer_names: Optional[List[str]] = None,
    head_names: Optional[List[str]] = None,
    title: str = "Attention Patterns",
    cmap: str = "viridis",
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Plot a grid of attention heatmaps (layers x heads).

    Args:
        attention_tensor: Array of shape (n_layers, n_heads, seq_len, seq_len)
        output_path: Where to save the figure
        layer_names: Optional list of names for layers (Y-axis)
        head_names: Optional list of names for heads (X-axis)
        title: Overall figure title
        cmap: Matplotlib colormap
        figsize: Figure size (width, height)
    """
    if not HAS_MATPLOTLIB:
        return

    n_layers, n_heads, seq_len, _ = attention_tensor.shape

    if figsize is None:
        figsize = (n_heads * 3, n_layers * 3)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=16)

    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            im = ax.imshow(attention_tensor[l, h], cmap=cmap, vmin=0, vmax=1)

            # Labels
            if l == n_layers - 1:
                h_name = head_names[h] if head_names else f"Head {h}"
                ax.set_xlabel(h_name)
            if h == 0:
                l_name = layer_names[l] if layer_names else f"Layer {l}"
                ax.set_ylabel(l_name)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_attention_entropy(
    steps: np.ndarray,
    entropy_history: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Attention Entropy vs. Training Step",
    labels: Optional[List[str]] = None
):
    """
    Plot attention entropy over training steps.

    Args:
        steps: Array of step numbers, shape (N,)
        entropy_history: Array of entropy values. Can be 1D (overall) or
            2D (N, n_lines) to plot multiple lines (e.g., per-head or per-layer).
        output_path: Where to save the figure
        title: Overall figure title
        labels: List of labels for each line if entropy_history is 2D
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if entropy_history.ndim == 1:
        ax.plot(steps, entropy_history, label="Mean Entropy", color='blue', linewidth=2)
    else:
        n_lines = entropy_history.shape[1]
        for i in range(n_lines):
            lbl = labels[i] if labels else f"Group {i}"
            ax.plot(steps, entropy_history[:, i], label=lbl, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.grid(True, alpha=0.3)

    if entropy_history.ndim > 1 or labels is not None:
        ax.legend()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_head_specialization_clusters(
    attention_tensor: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Head Specialization Clustering"
):
    """
    Plot a dendrogram clustering attention heads by their pattern similarity.

    Args:
        attention_tensor: Array of shape (n_layers, n_heads, seq_len, seq_len)
                          or (n_heads_total, seq_len, seq_len).
        output_path: Where to save the figure
        title: Overall figure title
    """
    if not HAS_MATPLOTLIB or not HAS_SCIPY:
        return

    if attention_tensor.ndim == 4:
        n_layers, n_heads, seq_len, _ = attention_tensor.shape
        flat_heads = attention_tensor.reshape(n_layers * n_heads, -1)
        labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    else:
        n_heads, seq_len, _ = attention_tensor.shape
        flat_heads = attention_tensor.reshape(n_heads, -1)
        labels = [f"Head {h}" for h in range(n_heads)]

    # If only 1 head, clustering is trivial and pdist fails/returns empty
    if len(labels) <= 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Only 1 head available; no clustering needed.",
                ha='center', va='center')
        ax.set_title(title)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    # Compute pairwise Euclidean distances between flattened head patterns
    distances = pdist(flat_heads, metric='euclidean')

    # Hierarchical clustering
    linkage_matrix = hierarchy.linkage(distances, method='ward')

    fig, ax = plt.subplots(figsize=(10, 6))
    hierarchy.dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90)

    ax.set_title(title)
    ax.set_ylabel("Ward Distance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def animate_attention_evolution(
    attention_history: np.ndarray,
    output_path: Union[str, Path],
    steps: Optional[np.ndarray] = None,
    fps: int = 5,
    title: str = "Attention Evolution"
):
    """
    Animate the evolution of an attention pattern over time.

    Args:
        attention_history: Array of shape (n_steps, seq_len, seq_len)
        output_path: Where to save the animation (should end in .gif or .mp4)
        steps: Array of step numbers corresponding to each frame
        fps: Frames per second
        title: Base title for the animation
    """
    if not HAS_MATPLOTLIB:
        return

    n_steps, seq_len, _ = attention_history.shape

    fig, ax = plt.subplots(figsize=(5, 5))

    # Initial plot
    im = ax.imshow(attention_history[0], cmap='viridis', vmin=0, vmax=1)

    title_obj = ax.set_title(f"{title} - Step {steps[0] if steps is not None else 0}")
    fig.colorbar(im, ax=ax)

    def update(frame):
        im.set_array(attention_history[frame])
        step_str = f"Step {steps[frame]}" if steps is not None else f"Frame {frame}"
        title_obj.set_text(f"{title} - {step_str}")
        return [im, title_obj]

    anim = animation.FuncAnimation(
        fig, update, frames=n_steps, interval=1000/fps, blit=True
    )

    out_path = Path(output_path)
    if out_path.suffix == '.gif':
        anim.save(out_path, writer='pillow', fps=fps)
    else:
        anim.save(out_path, fps=fps)

    plt.close(fig)
