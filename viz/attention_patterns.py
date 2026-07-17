"""
Attention pattern visualization for grokking-collapse experiments.
Generates static heatmaps and animated GIFs of attention weight evolution,
and compares patterns between pure and collapsed models.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ModularArithmeticTransformer

def load_model_from_checkpoint(ckpt_path: Path) -> ModularArithmeticTransformer:
    """Load model from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def extract_attention_weights(model: ModularArithmeticTransformer, inputs: torch.Tensor) -> torch.Tensor:
    """
    Extract attention weights for the given inputs.
    Returns tensor of shape (batch, n_heads, seq_len, seq_len).
    """
    with torch.no_grad():
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        # Token embeddings
        tok = model.token_embed(inputs)  # (batch, seq_len, d_model)

        # Positional embeddings
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)

        h = tok + pos

        # Extract attention weights using layer.self_attn
        layer = model.transformer.layers[0]
        # need_weights=True, average_attn_weights=False returns (attn_output, attn_weights)
        # where attn_weights is (batch_size, num_heads, tgt_len, src_len)
        _, attn_weights = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)

        return attn_weights

def plot_attention_heatmap(attn_weights: torch.Tensor, ax: plt.Axes, title: str = ""):
    """
    Plot attention heatmap for a single head.
    attn_weights: (seq_len, seq_len)
    """
    im = ax.imshow(attn_weights.numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_xticks(range(attn_weights.shape[0]))
    ax.set_yticks(range(attn_weights.shape[0]))
    return im

def plot_model_attention(model: ModularArithmeticTransformer, inputs: torch.Tensor,
                         save_path: Optional[Path] = None, title: str = "Attention Patterns"):
    """Plot attention heatmaps for all heads of the model."""
    attn_weights = extract_attention_weights(model, inputs)
    # Average across batch
    avg_attn = attn_weights.mean(dim=0)  # (n_heads, seq_len, seq_len)

    n_heads = avg_attn.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        plot_attention_heatmap(avg_attn[i], ax, f"Head {i+1}")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def create_attention_animation(checkpoint_paths: List[Path], inputs: torch.Tensor,
                               save_path: Path, title_prefix: str = ""):
    """Create an animated GIF of attention weights over training steps."""
    if not checkpoint_paths:
        return

    fig, axes = None, None
    n_heads = 0
    ims = []

    for ckpt_path in sorted(checkpoint_paths, key=lambda p: int(p.stem.split("_")[1])):
        model = load_model_from_checkpoint(ckpt_path)
        attn_weights = extract_attention_weights(model, inputs)
        avg_attn = attn_weights.mean(dim=0)  # (n_heads, seq_len, seq_len)
        step = int(ckpt_path.stem.split("_")[1])

        if fig is None:
            n_heads = avg_attn.shape[0]
            fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
            if n_heads == 1:
                axes = [axes]
            fig.suptitle(f"{title_prefix}Attention Evolution", fontsize=14)

        frame_artists = []
        for i, ax in enumerate(axes):
            im = ax.imshow(avg_attn[i].numpy(), cmap="viridis", vmin=0, vmax=1, animated=True)
            ax.set_title(f"Head {i+1}")
            title_text = ax.text(0.5, 1.15, f"Step: {step}", transform=ax.transAxes,
                                 ha="center", fontsize=12, animated=True)
            frame_artists.extend([im, title_text])
        ims.append(frame_artists)

    if fig and ims:
        plt.tight_layout()
        ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
        ani.save(save_path, writer='pillow', dpi=100)
        plt.close(fig)

def compare_pure_vs_collapsed(pure_ckpt: Path, collapsed_ckpt: Path, inputs: torch.Tensor, save_path: Path):
    """Compare attention patterns between pure and collapsed models."""
    pure_model = load_model_from_checkpoint(pure_ckpt)
    collapsed_model = load_model_from_checkpoint(collapsed_ckpt)

    pure_attn = extract_attention_weights(pure_model, inputs).mean(dim=0)
    collapsed_attn = extract_attention_weights(collapsed_model, inputs).mean(dim=0)

    n_heads = pure_attn.shape[0]
    fig, axes = plt.subplots(2, n_heads, figsize=(4 * n_heads, 8))
    if n_heads == 1:
        axes = np.array([axes])

    for i in range(n_heads):
        plot_attention_heatmap(pure_attn[i], axes[0, i], f"Pure Head {i+1}")
        plot_attention_heatmap(collapsed_attn[i], axes[1, i], f"Collapsed Head {i+1}")

    fig.suptitle("Attention Pattern Comparison (Pure vs Collapsed)", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate attention pattern visualizations")
    parser.add_argument("--results-dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--output-dir", type=str, default="viz_output", help="Path to output directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate some random inputs for visualization
    # using prime 59 as default
    inputs = torch.randint(0, 59, (10, 2))

    # 1. Plot pure model attention at last checkpoint
    pure_dir = results_dir / "pure"
    if pure_dir.exists():
        checkpoints = list(pure_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            last_ckpt = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
            model = load_model_from_checkpoint(last_ckpt)
            plot_model_attention(model, inputs, output_dir / "attention_pure_final.png", "Pure Model Attention (Final)")

            # Animation
            create_attention_animation(checkpoints, inputs, output_dir / "attention_evolution_pure.gif", "Pure ")

    # 2. Compare pure vs collapsed
    severe_dir = results_dir / "severe_collapse"
    if pure_dir.exists() and severe_dir.exists():
        pure_checkpoints = list(pure_dir.glob("checkpoint_*.pt"))
        severe_checkpoints = list(severe_dir.glob("checkpoint_*.pt"))

        if pure_checkpoints and severe_checkpoints:
            last_pure = max(pure_checkpoints, key=lambda p: int(p.stem.split("_")[1]))
            last_severe = max(severe_checkpoints, key=lambda p: int(p.stem.split("_")[1]))
            compare_pure_vs_collapsed(last_pure, last_severe, inputs, output_dir / "attention_compare.png")

            # Animation for collapsed
            create_attention_animation(severe_checkpoints, inputs, output_dir / "attention_evolution_severe.gif", "Severe Collapse ")
