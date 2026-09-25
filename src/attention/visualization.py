import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional

def plot_attention_heatmaps(
    attentions: np.ndarray,
    tokens: List[str],
    layer_idx: int,
    save_path: str,
    title: Optional[str] = None
):
    """
    Plot attention heatmaps for a single layer across all heads.

    Args:
        attentions: Array of shape (num_heads, seq_len, seq_len)
        tokens: List of token strings (length seq_len)
        layer_idx: The index of the layer
        save_path: Path to save the PNG file
        title: Optional title for the figure
    """
    num_heads, seq_len, _ = attentions.shape

    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attentions[h], cmap="Blues", vmin=0, vmax=1)

        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

        ax.set_title(f"Head {h}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for h in range(num_heads, len(axes)):
        axes[h].axis('off')

    if title is None:
        title = f"Attention Layer {layer_idx}"
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_attention_trajectory(
    steps: List[int],
    entropies: List[float],
    grokking_step: Optional[int],
    save_path: str,
    title: str = "Attention Entropy Trajectory"
):
    """
    Plot attention entropy trajectory across training steps.

    Args:
        steps: List of training steps
        entropies: List of entropy values corresponding to steps
        grokking_step: The step where grokking occurred, to overlay as a vertical line
        save_path: Path to save the PNG file
        title: Figure title
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(steps, entropies, label='Attention Entropy', color='#3498db', linewidth=2)

    if grokking_step is not None:
        ax.axvline(x=grokking_step, color='#e74c3c', linestyle='--',
                   label=f'Grokking Cliff (Step {grokking_step})', linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Shannon Entropy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_attention_diff(
    attn_a: np.ndarray,
    attn_b: np.ndarray,
    tokens: List[str],
    layer_idx: int,
    save_path: str,
    title_a: str = "Model A",
    title_b: str = "Model B"
):
    """
    Plot the difference between two attention matrices.

    Args:
        attn_a: Attention array for model A (num_heads, seq_len, seq_len)
        attn_b: Attention array for model B (num_heads, seq_len, seq_len)
        tokens: Token strings
        layer_idx: Layer index
        save_path: Path to save the PNG
        title_a: Label for model A
        title_b: Label for model B
    """
    # Diff = A - B
    diff = attn_a - attn_b
    num_heads, seq_len, _ = diff.shape

    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        # Diff usually ranges from -1 to 1
        im = ax.imshow(diff[h], cmap="RdBu_r", vmin=-1, vmax=1)

        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

        ax.set_title(f"Head {h}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for h in range(num_heads, len(axes)):
        axes[h].axis('off')

    fig.suptitle(f"Attention Diff ({title_a} - {title_b}) Layer {layer_idx}", fontsize=16)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_critical_token_trajectory(
    steps: List[int],
    attention_to_critical: np.ndarray,
    critical_token_idx: int,
    tokens: List[str],
    grokking_step: Optional[int],
    save_path: str,
    title: str = "Attention to Critical Token Trajectory"
):
    """
    Plot the trajectory of attention directed to a specific critical token for each head.

    Args:
        steps: List of training steps
        attention_to_critical: Array of shape (num_steps, num_heads) containing the attention
                               probability directed to the critical token.
        critical_token_idx: The index of the critical token.
        tokens: The token labels.
        grokking_step: Step where grokking occurred.
        save_path: Path to save the PNG file.
        title: Figure title.
    """
    num_steps, num_heads = attention_to_critical.shape
    fig, ax = plt.subplots(figsize=(10, 6))

    for h in range(num_heads):
        ax.plot(steps, attention_to_critical[:, h], label=f'Head {h}', linewidth=2, alpha=0.8)

    if grokking_step is not None:
        ax.axvline(x=grokking_step, color='#e74c3c', linestyle='--',
                   label=f'Grokking Cliff (Step {grokking_step})', linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(f"Attention Probability to '{tokens[critical_token_idx]}'")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
