import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional

def generate_attention_heatmap(attention_weights: torch.Tensor, output_path: Path):
    """Generate a static heatmap of attention weights."""
    fig, ax = plt.subplots(figsize=(6, 6))
    weights = attention_weights.detach().cpu().numpy()

    # Assuming shape (n_heads, seq_len, seq_len)
    if len(weights.shape) == 3:
        # Average across heads for a simple visualization
        weights = weights.mean(axis=0)

    cax = ax.matshow(weights, cmap='viridis')
    fig.colorbar(cax)

    plt.title("Attention Heatmap")

    # Save as PNG
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    # Save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_attention_animation(attention_weights_history: List[torch.Tensor], output_path: Path):
    """Generate a GIF animation of attention weights evolving over time."""
    # matplotlib text.usetex = False is default, avoiding ffmpeg TeX issues

    if not attention_weights_history:
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    def get_weights(idx):
        weights = attention_weights_history[idx].detach().cpu().numpy()
        if len(weights.shape) == 3:
            weights = weights.mean(axis=0)
        return weights

    im = ax.matshow(get_weights(0), cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(im)

    def update(frame):
        im.set_array(get_weights(frame))
        ax.set_title(f"Attention Heatmap - Step {frame}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(attention_weights_history), blit=True)

    # Save as GIF
    gif_path = output_path.with_suffix('.gif')
    ani.save(gif_path, writer='pillow', fps=5)
    plt.close()
