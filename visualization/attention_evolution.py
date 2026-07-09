import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.model import ModularArithmeticTransformer

def load_model_checkpoint(filepath: str) -> Tuple[ModularArithmeticTransformer, Dict]:
    """Load a model checkpoint and return the instantiated model and checkpoint dict."""
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
    except Exception:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

    config = checkpoint['config']

    model = ModularArithmeticTransformer(
        prime=config.get('prime', 59),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        d_ff=config.get('d_ff', 512),
        n_layers=config.get('n_layers', 1),
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint

def extract_attention_weights(model: ModularArithmeticTransformer, inputs: torch.Tensor) -> torch.Tensor:
    """
    Extract attention weights from the first transformer layer.

    Args:
        model: The ModularArithmeticTransformer model
        inputs: Input tensor of shape (batch, seq_len)

    Returns:
        Attention weights of shape (batch, n_heads, seq_len, seq_len)
    """
    batch_size, seq_len = inputs.shape

    # Token embeddings
    tok = model.token_embed(inputs)

    # Positional embeddings
    positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)

    # Combine
    h = tok + pos

    layer = model.transformer.layers[0]

    # Handle batch_first correctly
    is_batch_first = getattr(layer, 'batch_first', False)
    if not is_batch_first:
        # Standard PyTorch TransformerEncoderLayer expects (seq_len, batch, d_model) if batch_first is False
        h = h.transpose(0, 1)

    attn_output, attn_weights = layer.self_attn(
        h, h, h,
        need_weights=True,
        average_attn_weights=False
    )

    return attn_weights

def compute_attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of attention distributions over the target sequence.

    Args:
        attn_weights: Attention weights of shape (..., seq_len)

    Returns:
        Entropy of shape (...)
    """
    # Clamp to avoid log(0)
    p = torch.clamp(attn_weights, min=1e-10)
    entropy = -torch.sum(p * torch.log(p), dim=-1)
    return entropy

def compute_attention_similarity(attn1: torch.Tensor, attn2: torch.Tensor) -> float:
    """
    Compute cosine similarity between flattened attention matrices.
    """
    a1_flat = attn1.flatten()
    a2_flat = attn2.flatten()
    cos_sim = F.cosine_similarity(a1_flat.unsqueeze(0), a2_flat.unsqueeze(0))
    return cos_sim.item()

def plot_attention_heatmaps(attn_weights: torch.Tensor, step: int, save_path: str):
    """
    Plot attention heatmaps for all heads side-by-side.

    Args:
        attn_weights: (batch, n_heads, seq_len, seq_len) - average over batch or select first
        step: current training step
        save_path: where to save the image
    """
    # Average over batch
    avg_attn = attn_weights.mean(dim=0).detach().cpu().numpy()
    n_heads = avg_attn.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(avg_attn[i], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Head {i + 1}')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Attention Patterns at Step {step}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def animate_attention_evolution(checkpoints_paths: List[str], inputs: torch.Tensor, save_path: str):
    """
    Create an animation of attention patterns evolving across checkpoints.

    Args:
        checkpoints_paths: List of paths to model checkpoints
        inputs: Input tensor to evaluate attention on
        save_path: Path to save the animation (.mp4 or .gif)
    """
    # Load all attention weights first
    all_attn_weights = []
    steps = []

    for path in checkpoints_paths:
        model, ckpt = load_model_checkpoint(path)
        with torch.no_grad():
            attn = extract_attention_weights(model, inputs)
            # Average over batch
            attn = attn.mean(dim=0).cpu().numpy()
        all_attn_weights.append(attn)
        steps.append(ckpt.get('step', 0))

    if not all_attn_weights:
        print("No checkpoints provided.")
        return

    n_heads = all_attn_weights[0].shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    ims = []

    for i, ax in enumerate(axes):
        ax.set_title(f'Head {i + 1}')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')

    # Create the frames
    for idx, (attn, step) in enumerate(zip(all_attn_weights, steps)):
        frame_artists = []

        # Add title as an artist
        title = fig.text(0.5, 0.98, f'Attention Patterns at Step {step}',
                         ha='center', va='top', fontsize=14)
        frame_artists.append(title)

        for i, ax in enumerate(axes):
            im = ax.imshow(attn[i], cmap='viridis', vmin=0, vmax=1, animated=True)
            frame_artists.append(im)

            # Add colorbar on first frame
            if idx == 0:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ims.append(frame_artists)

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path.endswith('.mp4'):
        ani.save(save_path, writer='ffmpeg', dpi=150)
    elif save_path.endswith('.gif'):
        ani.save(save_path, writer='pillow', dpi=150)
    else:
        # Default to mp4 if unknown extension
        ani.save(save_path + '.mp4', writer='ffmpeg', dpi=150)

    plt.close(fig)
