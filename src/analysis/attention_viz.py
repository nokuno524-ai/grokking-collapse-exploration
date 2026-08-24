"""
Functions for calculating attention pattern metrics and generating visualization figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from typing import Union, List

def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Computes Shannon entropy of the attention distribution.

    Args:
        attn_weights: Tensor of shape (B, n_heads, T, T). Sums to 1 over the last dimension.
        eps: Small value to avoid log(0).

    Returns:
        entropy: Tensor of shape (B, n_heads, T) representing entropy for each query position.
    """
    # Entropy = - sum(p * log(p))
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
    return entropy

def compute_head_specialization(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Computes head specialization as the variance of attention patterns across heads.

    Higher variance means heads are attending to different things (more specialized).
    Lower variance means heads are redundant.

    Args:
        attn_weights: Tensor of shape (B, n_heads, T, T)

    Returns:
        specialization: Tensor of shape (B, T, T) representing variance across heads.
    """
    # Variance across the n_heads dimension
    return torch.var(attn_weights, dim=1)

def plot_attention_heatmap(attn_weights: Union[torch.Tensor, List[torch.Tensor]], layer_idx: int = 0, batch_idx: int = 0, head_idx: int = 0) -> plt.Figure:
    """
    Plots the attention heatmap for a specific layer, batch, and head.

    Args:
        attn_weights: List of tensors of shape (B, n_heads, T, T) for each layer, or single tensor
        layer_idx: Index of the transformer layer to visualize.
        batch_idx: Index of the item in the batch.
        head_idx: Index of the attention head.

    Returns:
        fig: matplotlib Figure object.
    """
    if isinstance(attn_weights, list):
        if layer_idx >= len(attn_weights):
            raise ValueError(f"layer_idx {layer_idx} out of bounds for {len(attn_weights)} layers.")
        attn_w = attn_weights[layer_idx]
    else:
        attn_w = attn_weights

    attn = attn_w[batch_idx, head_idx].cpu().numpy()
    T = attn.shape[0]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(attn, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_title(f"Layer {layer_idx} - Attention Head {head_idx}")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    # Tick marks for sequence length
    ax.set_xticks(np.arange(T))
    ax.set_yticks(np.arange(T))
    ax.set_xticklabels([f"Pos {i}" for i in range(T)])
    ax.set_yticklabels([f"Pos {i}" for i in range(T)])

    plt.tight_layout()
    return fig

def plot_attention_entropy_trajectory(attention_data: dict, layer_idx: int = 0) -> plt.Figure:
    """
    Plots average attention entropy over training steps for different conditions.

    Args:
        attention_data: Dict of form {condition: {step: list_of_attn_weights_tensors}}
        layer_idx: Which layer's entropy to plot.

    Returns:
        fig: matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for i, (condition, steps_dict) in enumerate(attention_data.items()):
        steps = sorted(list(steps_dict.keys()))
        entropies = []

        for step in steps:
            attn_item = steps_dict[step]
            if isinstance(attn_item, list):
                attn = attn_item[layer_idx]
            else:
                attn = attn_item # fallback for old extracted data format

            # Compute mean entropy over batch, heads, and query positions
            ent = compute_attention_entropy(attn).mean().item()
            entropies.append(ent)

        ax.plot(steps, entropies, marker=markers[i%len(markers)], color=colors[i%len(colors)],
                label=condition, linewidth=2, markersize=6)

    ax.set_title(f"Attention Entropy over Training (Layer {layer_idx})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Attention Entropy (nats)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def plot_head_specialization_trajectory(attention_data: dict, layer_idx: int = 0) -> plt.Figure:
    """
    Plots average head specialization (variance) over training steps for different conditions.

    Args:
        attention_data: Dict of form {condition: {step: list_of_attn_weights_tensors}}
        layer_idx: Which layer's specialization to plot.

    Returns:
        fig: matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for i, (condition, steps_dict) in enumerate(attention_data.items()):
        steps = sorted(list(steps_dict.keys()))
        specializations = []

        for step in steps:
            attn_item = steps_dict[step]
            if isinstance(attn_item, list):
                attn = attn_item[layer_idx]
            else:
                attn = attn_item

            # Compute mean specialization (variance) over batch and spatial dimensions
            spec = compute_head_specialization(attn).mean().item()
            specializations.append(spec)

        ax.plot(steps, specializations, marker=markers[i%len(markers)], color=colors[i%len(colors)],
                label=condition, linewidth=2, markersize=6)

    ax.set_title(f"Head Specialization over Training (Layer {layer_idx})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Cross-Head Variance")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig
