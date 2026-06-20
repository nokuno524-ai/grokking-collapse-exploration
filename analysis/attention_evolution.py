"""
Attention Pattern Evolution Tracker.
Loads model checkpoints at intervals, extracts attention weights from all layers/heads,
computes per-head attention entropy, tracks head specialization, and generates visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.model import ModularArithmeticTransformer

def extract_attention_weights(model: ModularArithmeticTransformer, x: torch.Tensor) -> list[torch.Tensor]:
    """
    Extract attention weights from the model for the given input.
    """
    model.eval()
    with torch.no_grad():
        snapshots = model.get_attention_snapshots(x)
    return snapshots

def compute_attention_entropy(attn_weights: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Compute Shannon entropy of attention weights per head.
    attn_weights shape: (batch, n_heads, seq_len, seq_len)
    """
    # Normalize if not exactly summing to 1 due to float precision
    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + epsilon)
    # Entropy over the key dimension (last dimension)
    entropy = - (attn_weights * torch.log(attn_weights + epsilon)).sum(dim=-1)
    # Average over sequence length and batch
    return entropy.mean(dim=(0, 2))

def analyze_attention_evolution(checkpoint_dir: str | Path, model_config: dict, sample_inputs: torch.Tensor):
    """
    Analyze attention evolution across training checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoint_*.pt files.
        model_config: Dictionary of model kwargs to initialize the model.
        sample_inputs: Inputs to run through the model for attention extraction.

    Returns:
        dict: A dictionary of metrics tracked over time.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(
        [p for p in checkpoint_dir.glob("checkpoint_*.pt")],
        key=lambda p: int(p.stem.split('_')[1])
    )

    metrics = {
        'steps': [],
        'entropy_per_head': [],  # list of (n_layers, n_heads) tensors
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModularArithmeticTransformer(**model_config).to(device)
    sample_inputs = sample_inputs.to(device)

    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.split('_')[1])
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state'])
        except Exception:
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state'])
            except Exception as e:
                print(f"Skipping {ckpt_path}: {e}")
                continue

        snapshots = extract_attention_weights(model, sample_inputs)

        # Calculate entropy per head per layer
        # list of tensors of shape (n_heads,)
        layer_entropies = [compute_attention_entropy(attn) for attn in snapshots]
        layer_entropies = torch.stack(layer_entropies).cpu().numpy()  # (n_layers, n_heads)

        metrics['steps'].append(step)
        metrics['entropy_per_head'].append(layer_entropies)

    if metrics['entropy_per_head']:
        metrics['entropy_per_head'] = np.stack(metrics['entropy_per_head'])  # (n_checkpoints, n_layers, n_heads)

    return metrics

def plot_attention_entropy(metrics: dict, output_path: str | Path):
    """Plot attention entropy over time."""
    steps = metrics['steps']
    entropy = metrics['entropy_per_head']  # (n_checkpoints, n_layers, n_heads)

    if len(steps) == 0:
        print("No metrics to plot.")
        return

    n_layers, n_heads = entropy.shape[1], entropy.shape[2]

    fig, axes = plt.subplots(n_layers, 1, figsize=(10, 4 * n_layers), squeeze=False)

    for l in range(n_layers):
        ax = axes[l, 0]
        for h in range(n_heads):
            ax.plot(steps, entropy[:, l, h], label=f'Head {h}')
        ax.set_title(f'Layer {l} Attention Entropy')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Entropy')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_attention_heatmap(model: ModularArithmeticTransformer, sample_inputs: torch.Tensor, output_path: str | Path):
    """Plot attention heatmaps for all layers and heads for the first input in batch."""
    snapshots = extract_attention_weights(model, sample_inputs)
    # snapshots: list of (batch, n_heads, seq_len, seq_len)

    n_layers = len(snapshots)
    n_heads = snapshots[0].size(1)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers), squeeze=False)

    # We'll just plot the first item in the batch
    idx = 0

    for l in range(n_layers):
        attn = snapshots[l][idx].cpu().numpy()  # (n_heads, seq_len, seq_len)
        for h in range(n_heads):
            ax = axes[l, h]
            sns.heatmap(attn[h], ax=ax, vmin=0, vmax=1, cmap="YlGnBu", cbar=(h == n_heads - 1))
            ax.set_title(f'Layer {l} Head {h}')
            ax.set_xlabel('Key Pos')
            ax.set_ylabel('Query Pos')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
