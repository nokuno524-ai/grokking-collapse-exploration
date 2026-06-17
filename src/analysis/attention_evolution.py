"""
Attention Pattern Evolution Tracking
Extracts attention patterns from the model to calculate entropy and visualize pattern shift over training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import math


def extract_attention_patterns(model: nn.Module, dataloader, device: torch.device) -> torch.Tensor:
    """
    Extracts attention weights from the TransformerEncoderLayer.
    Returns: Tensor of shape (num_samples, n_heads, seq_len, seq_len)
    """
    model.eval()
    all_attentions = []

    # We will temporarily mock the forward pass of the transformer layer
    # to explicitly extract attention weights, since nn.TransformerEncoderLayer
    # doesn't store attention weights by default unless manually handled or hooked.

    layer = model.transformer.layers[0] if hasattr(model.transformer, 'layers') else list(model.transformer.children())[0]

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.shape[0]

            seq_len = inputs.shape[1]
            tok = model.token_embed(inputs)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos = model.pos_embed(positions)
            h = tok + pos

            # Apply Pre-LN if present (or standard LN), model may not have it explicitly before attn,
            # but ModularArithmeticTransformer uses post-LN in its layers.
            # TransformerEncoderLayer applies its own norm1 inside. Let's replicate norm1.
            # norm1 is applied BEFORE self_attn if norm_first=True, else AFTER. By default in PyTorch it's AFTER.
            if hasattr(layer, 'norm_first') and layer.norm_first:
                h_norm = layer.norm1(h)
            else:
                h_norm = h # Standard PyTorch is post-LN, so query goes in un-normalized usually, or via standard forward.

            if getattr(layer.self_attn, 'batch_first', False):
                query = h_norm
            else:
                query = h_norm.transpose(0, 1)

            attn_output, attn_weights = layer.self_attn(
                query, query, query,
                need_weights=True,
                average_attn_weights=False
            )

            # attn_weights shape: (batch_size, n_heads, seq_len, seq_len)
            all_attentions.append(attn_weights.cpu())

    return torch.cat(all_attentions, dim=0)


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention weights per head.
    attention_weights: (batch_size, n_heads, seq_len, seq_len)
    Returns: (batch_size, n_heads, seq_len)
    """
    # Entropy = - sum(p * log(p))
    # Add epsilon to prevent log(0)
    eps = 1e-10
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1)
    return entropy


def track_attention_evolution(models_dict: Dict[str, nn.Module], dataloader, device: torch.device):
    """
    Compute mean attention entropy across stages.
    models_dict: {'pre': model1, 'grok': model2, 'post': model3}
    Returns average entropy per head.
    """
    results = {}

    for phase, model in models_dict.items():
        attn_weights = extract_attention_patterns(model, dataloader, device)
        entropy = compute_attention_entropy(attn_weights)
        # Average over batch and seq_len
        mean_entropy = entropy.mean(dim=(0, 2))
        results[phase] = mean_entropy

    return results

def plot_attention_heatmaps(attention_weights_dict: Dict[str, torch.Tensor], output_path: str):
    """
    Generate attention evolution visualizations: heatmap sequences showing pattern change.
    attention_weights_dict: phase -> attention tensor of shape (batch, n_heads, seq_len, seq_len)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    phases = list(attention_weights_dict.keys())
    if not phases:
        return

    n_heads = attention_weights_dict[phases[0]].shape[1]

    fig, axes = plt.subplots(len(phases), n_heads, figsize=(4 * n_heads, 4 * len(phases)))
    if len(phases) == 1:
        axes = [axes]

    for i, phase in enumerate(phases):
        # Average over batch
        attn = attention_weights_dict[phase].mean(dim=0).cpu().numpy()

        for h in range(n_heads):
            ax = axes[i][h] if n_heads > 1 else axes[i]
            sns.heatmap(attn[h], ax=ax, cmap="viridis", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(f"Head {h}")
            if h == 0:
                ax.set_ylabel(phase)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def identify_collapse_affected_heads(pure_entropy: torch.Tensor, collapsed_entropy: torch.Tensor) -> Tuple[List[int], List[int]]:
    """
    Identify which attention heads are most/least affected by model collapse.
    Returns (most_affected_heads, least_affected_heads) as lists of indices.
    """
    # Entropy diff: shape (n_heads,)
    diff = torch.abs(pure_entropy - collapsed_entropy)

    sorted_indices = torch.argsort(diff, descending=True)
    n_heads = len(diff)

    most_affected = sorted_indices[:n_heads//2].tolist()
    least_affected = sorted_indices[n_heads//2:].tolist()

    return most_affected, least_affected
