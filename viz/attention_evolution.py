import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math
import os
import sys

# Add the project root to the sys path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ModularArithmeticTransformer

def get_attention_patterns(model: ModularArithmeticTransformer, x: torch.Tensor) -> torch.Tensor:
    """
    Manually compute the attention weights for the first layer, as nn.TransformerEncoderLayer
    does not expose them by default.
    """
    model.eval()
    with torch.no_grad():
        batch_size = x.shape[0]

        # Token embeddings
        tok = model.token_embed(x)  # (batch, 2, d_model)

        # Positional embeddings
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)  # (batch, 2, d_model)

        # Combine
        h = tok + pos  # (batch, 2, d_model)

        # Extract parameters for the first layer's self-attention
        layer = model.transformer.layers[0]
        qkv_weight = layer.self_attn.in_proj_weight
        qkv_bias = layer.self_attn.in_proj_bias

        # In PyTorch, if batch_first=True in nn.MultiheadAttention, input is (B, L, E)
        # However, nn.TransformerEncoderLayer applies attention, but let's just project manually
        # since we want the raw attention weights.

        # Project Q, K, V
        d_model = model.d_model
        n_heads = model.n_heads
        head_dim = d_model // n_heads

        qkv = F.linear(h, qkv_weight, qkv_bias) # (B, L, 3 * E)

        q, k, v = qkv.chunk(3, dim=-1) # Each is (B, L, E)

        # Reshape for multi-head attention: (B, L, n_heads, head_dim) -> (B, n_heads, L, head_dim)
        q = q.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim) # (B, n_heads, L, L)

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)

        return attn_weights

def load_attention_weights(checkpoint_path: str, dummy_input: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Load a checkpoint, instantiate the model, and compute attention weights for a given input.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint.get('config', {})

    # Instantiate model with config from checkpoint (fallback to defaults)
    model = ModularArithmeticTransformer(
        prime=config.get('prime', 59),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        d_ff=config.get('d_ff', 512),
        n_layers=config.get('n_layers', 1)
    )
    model.load_state_dict(checkpoint['model_state'])

    if dummy_input is None:
        dummy_input = torch.randint(0, config.get('prime', 59), (4, 2))

    attn_weights = get_attention_patterns(model, dummy_input)

    # Return averaged across batch
    return attn_weights.mean(dim=0) # (n_heads, L, L)

def plot_attention_heatmaps(attn_weights: torch.Tensor, title: str = "Attention Patterns", save_path: Optional[str] = None):
    """
    Plot attention heatmaps for all heads in the first layer.
    """
    n_heads = attn_weights.shape[0]

    # Calculate grid size
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for h in range(n_heads):
        ax = axes[h]
        sns.heatmap(attn_weights[h].numpy(), ax=ax, cmap="Blues", vmin=0, vmax=1,
                    xticklabels=["a", "b"], yticklabels=["a", "b"], annot=True, fmt=".2f")
        ax.set_title(f"Head {h+1}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    for h in range(n_heads, len(axes)):
        axes[h].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig

def generate_time_evolution_grid(checkpoint_paths: List[str], steps: List[int], save_path: str, dummy_input: Optional[torch.Tensor] = None):
    """
    Generate a grid of attention heatmaps across training steps.
    Rows: Checkpoint steps
    Cols: Attention heads
    """
    n_steps = len(checkpoint_paths)
    if n_steps == 0:
        print("No checkpoints provided.")
        return

    # Get first checkpoint to know number of heads
    attn_first = load_attention_weights(checkpoint_paths[0], dummy_input)
    n_heads = attn_first.shape[0]

    fig, axes = plt.subplots(n_steps, n_heads, figsize=(n_heads * 3, n_steps * 3))

    if n_steps == 1:
        axes = np.array([axes])
    if n_heads == 1:
        axes = axes[:, np.newaxis]

    for i, (path, step) in enumerate(zip(checkpoint_paths, steps)):
        attn_weights = load_attention_weights(path, dummy_input)

        for h in range(n_heads):
            ax = axes[i, h]
            sns.heatmap(attn_weights[h].numpy(), ax=ax, cmap="Blues", vmin=0, vmax=1,
                        xticklabels=["a", "b"], yticklabels=["a", "b"])

            if i == 0:
                ax.set_title(f"Head {h+1}")
            if h == 0:
                ax.set_ylabel(f"Step {step}\nQuery")
            else:
                ax.set_ylabel("Query")
            if i == n_steps - 1:
                ax.set_xlabel("Key")

    plt.suptitle("Attention Pattern Evolution over Training")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Time evolution grid saved to {save_path}")
    return fig

def compare_attention_evolution_side_by_side(run_pure_paths: List[str], run_collapsed_paths: List[str],
                                            steps: List[int], save_path: str, dummy_input: Optional[torch.Tensor] = None):
    """
    Generate a side-by-side time evolution grid comparing pure vs collapsed conditions.
    """
    n_steps = len(steps)
    if n_steps == 0:
        return

    attn_first = load_attention_weights(run_pure_paths[0], dummy_input)
    n_heads = attn_first.shape[0]

    # 2 columns per head (one for pure, one for collapsed)
    fig, axes = plt.subplots(n_steps, n_heads * 2, figsize=(n_heads * 6, n_steps * 3))

    if n_steps == 1:
        axes = np.array([axes])

    for i, step in enumerate(steps):
        attn_pure = load_attention_weights(run_pure_paths[i], dummy_input)
        attn_collapsed = load_attention_weights(run_collapsed_paths[i], dummy_input)

        for h in range(n_heads):
            # Pure
            ax_pure = axes[i, h*2]
            sns.heatmap(attn_pure[h].numpy(), ax=ax_pure, cmap="Blues", vmin=0, vmax=1,
                        xticklabels=["a", "b"], yticklabels=["a", "b"])

            # Collapsed
            ax_collapsed = axes[i, h*2 + 1]
            sns.heatmap(attn_collapsed[h].numpy(), ax=ax_collapsed, cmap="Reds", vmin=0, vmax=1,
                        xticklabels=["a", "b"], yticklabels=["a", "b"])

            if i == 0:
                ax_pure.set_title(f"Pure Head {h+1}")
                ax_collapsed.set_title(f"Collapsed Head {h+1}")
            if h == 0:
                ax_pure.set_ylabel(f"Step {step}\nQuery")

    plt.suptitle("Attention Pattern Evolution: Pure vs Collapsed")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

if __name__ == "__main__":
    # Test script with dummy checkpoint
    checkpoint_path = "tests/data/dummy_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print("Testing attention pattern generation...")
        attn = load_attention_weights(checkpoint_path)
        print(f"Loaded attention weights shape: {attn.shape}")

        os.makedirs("tests/output", exist_ok=True)
        fig = plot_attention_heatmaps(attn, save_path="tests/output/dummy_attention.png")
        print("Saved test heatmap to tests/output/dummy_attention.png")

        generate_time_evolution_grid([checkpoint_path, checkpoint_path], [0, 1000], "tests/output/dummy_evolution.png")
    else:
        print("Run tests/generate_checkpoint.py first to create a dummy checkpoint.")
