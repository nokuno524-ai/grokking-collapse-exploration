import torch
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import sys

# Add src to python path to import model
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from model import ModularArithmeticTransformer

def load_model_and_extract_attention(checkpoint_path: str, input_data: torch.Tensor, device: torch.device = torch.device('cpu')) -> Tuple[ModularArithmeticTransformer, torch.Tensor]:
    """
    Loads a checkpoint and extracts attention matrices for a given input.

    Args:
        checkpoint_path: Path to the .pt or .pth checkpoint file.
        input_data: Input tensor of shape (batch, 2).
        device: Device to load the model on.

    Returns:
        Tuple of (loaded model, attention weights tensor).
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        # Fallback for old checkpoints if weights_only fails
        ckpt = torch.load(checkpoint_path, map_location=device)

    config = ckpt.get('config', {})

    # Initialize model with config
    model = ModularArithmeticTransformer(
        prime=config.get('prime', 59),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        d_ff=config.get('d_ff', 512),
        n_layers=config.get('n_layers', 1),
    )
    model.to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Extract attention weights
    # We bypass the full forward pass to directly access the transformer layer
    with torch.no_grad():
        tok = model.token_embed(input_data.to(device))
        batch_size = input_data.shape[0]
        positions = torch.arange(2, device=device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)
        h = tok + pos

        # Access the first (and only) transformer layer
        layer = model.transformer.layers[0]

        # Extract attention weights by calling self_attn directly
        # need_weights=True, average_attn_weights=False gives per-head weights
        # Shape: (batch, num_heads, seq_len, seq_len)
        attn_output, attn_weights = layer.self_attn(
            h, h, h,
            need_weights=True,
            average_attn_weights=False
        )

    return model, attn_weights

if __name__ == "__main__":
    pass

def plot_attention_heatmap_grids(attn_weights_list: List[torch.Tensor], steps: List[int], output_path: str):
    """
    Plots a grid of attention heatmaps showing evolution across training steps.

    Args:
        attn_weights_list: List of attention weight tensors, each shape (batch, num_heads, seq_len, seq_len)
        steps: List of training steps corresponding to each tensor
        output_path: Path to save the figure
    """
    if not attn_weights_list:
        return

    num_steps = len(steps)
    num_heads = attn_weights_list[0].shape[1]

    fig, axes = plt.subplots(num_heads, num_steps, figsize=(4*num_steps, 4*num_heads))

    # Handle case where num_heads or num_steps is 1
    if num_heads == 1 and num_steps == 1:
        axes = np.array([[axes]])
    elif num_heads == 1:
        axes = axes[np.newaxis, :]
    elif num_steps == 1:
        axes = axes[:, np.newaxis]

    for step_idx, (attn_weights, step) in enumerate(zip(attn_weights_list, steps)):
        # Average across batch
        # Shape: (num_heads, seq_len, seq_len)
        avg_attn = attn_weights.mean(dim=0).cpu().numpy()

        for head_idx in range(num_heads):
            ax = axes[head_idx, step_idx]
            im = ax.imshow(avg_attn[head_idx], cmap='viridis', vmin=0, vmax=1)

            if step_idx == 0:
                ax.set_ylabel(f'Head {head_idx+1}')
            if head_idx == 0:
                ax.set_title(f'Step {step}')

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['a', 'b'])
            ax.set_yticklabels(['a', 'b'])

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def compute_attention_rollout(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Computes attention rollout (Abnar & Zhou 2020).
    For a 1-layer model, this is just the average attention plus identity, normalized.

    Args:
        attn_weights: Attention weights of shape (batch, num_heads, seq_len, seq_len)

    Returns:
        Rollout matrix of shape (batch, seq_len, seq_len)
    """
    # Average across heads: (batch, seq_len, seq_len)
    avg_attn = attn_weights.mean(dim=1)

    batch_size, seq_len, _ = avg_attn.shape
    device = avg_attn.device

    # Add identity matrix to simulate residual connection
    identity = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    rollout = avg_attn + identity

    # Normalize rows
    rollout = rollout / rollout.sum(dim=-1, keepdim=True)

    return rollout

def compute_attention_entropy(attn_weights: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Computes Shannon entropy of attention distributions per head.

    Args:
        attn_weights: Attention weights of shape (batch, num_heads, seq_len, seq_len)
        epsilon: Small value to prevent log(0)

    Returns:
        Entropy tensor of shape (batch, num_heads, seq_len)
    """
    # Entropy = -sum(p * log(p))
    entropy = -(attn_weights * torch.log(attn_weights + epsilon)).sum(dim=-1)
    return entropy

def plot_overlay_comparison(metrics_dict: Dict[str, Dict[str, List[float]]], metric_name: str, output_path: str):
    """
    Plots overlay comparison of attention metrics across different conditions.

    Args:
        metrics_dict: Dict of form {condition_name: {step: metric_value}}
        metric_name: Name of metric (e.g. 'Attention Entropy')
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    for condition, steps_dict in metrics_dict.items():
        steps = sorted(list(steps_dict.keys()))
        values = [steps_dict[step] for step in steps]
        plt.plot(steps, values, label=condition, marker='o', markersize=4)

    plt.xlabel('Training Steps')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Evolution Across Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
