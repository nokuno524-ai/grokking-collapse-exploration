import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from typing import List, Dict, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ModularArithmeticTransformer

def load_checkpoint(filepath: str, device: str = 'cpu') -> tuple:
    """Load model state and config from checkpoint."""
    ckpt = torch.load(filepath, map_location=device, weights_only=True)
    config = ckpt.get('config', {})

    # Instantiate model
    prime = config.get('prime', 59)
    d_model = config.get('d_model', 128)
    n_heads = config.get('n_heads', 4)
    d_ff = config.get('d_ff', 512)
    n_layers = config.get('n_layers', 1)

    model = ModularArithmeticTransformer(
        prime=prime, d_model=d_model, n_heads=n_heads,
        d_ff=d_ff, n_layers=n_layers
    )

    # Load state dict
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    step = ckpt.get('step', -1)
    return model, config, step

def compute_attention_patterns(model: ModularArithmeticTransformer, prime: int = 59):
    """
    Manually compute Q, K, V and attention patterns for all pairs of inputs (a, b).
    Returns attention from position 1 (b) to position 0 (a) for each head.
    Shape: (n_heads, prime, prime)
    """
    device = next(model.parameters()).device

    # Create all pairs (a, b)
    a = torch.arange(prime, device=device)
    b = torch.arange(prime, device=device)
    A, B = torch.meshgrid(a, b, indexing='ij')
    x = torch.stack([A.flatten(), B.flatten()], dim=1) # (prime^2, 2)

    with torch.no_grad():
        tok = model.token_embed(x)
        pos = model.pos_embed(torch.arange(2, device=device).unsqueeze(0).expand(x.shape[0], -1))
        h = tok + pos # (prime^2, 2, d_model)

        # Get attention weights
        # h is input to self-attention
        in_proj_weight = model.transformer.layers[0].self_attn.in_proj_weight
        in_proj_bias = model.transformer.layers[0].self_attn.in_proj_bias

        # qkv: (prime^2, 2, 3 * d_model)
        qkv = F.linear(h, in_proj_weight, in_proj_bias)

        # Reshape to separate q, k, v and heads
        # qkv shape: (prime^2, 2, 3, n_heads, head_dim)
        n_heads = model.n_heads
        head_dim = model.d_model // n_heads

        qkv = qkv.reshape(x.shape[0], 2, 3, n_heads, head_dim)

        q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3) # (prime^2, n_heads, 2, head_dim)
        k = qkv[:, :, 1, :, :].permute(0, 2, 1, 3) # (prime^2, n_heads, 2, head_dim)

        # Compute attention scores
        # scores = q @ k^T / sqrt(head_dim)
        # q: (prime^2, n_heads, 2, head_dim)
        # k.transpose(-2, -1): (prime^2, n_heads, head_dim, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5) # (prime^2, n_heads, 2, 2)

        # Apply softmax to get probabilities
        attn = F.softmax(scores, dim=-1) # (prime^2, n_heads, 2, 2)

        # We want to see how much position 1 attends to position 0
        # attn[:, :, 1, 0] is attention from pos 1 to pos 0
        attn_1_to_0 = attn[:, :, 1, 0] # (prime^2, n_heads)

        # Reshape to grid
        attn_grid = attn_1_to_0.reshape(prime, prime, n_heads).permute(2, 0, 1) # (n_heads, prime, prime)

    return attn_grid.cpu().numpy()

def generate_attention_heatmap(attn_grid: np.ndarray, step: int, save_path: str, title_suffix: str = ""):
    """Generate and save a heatmap for attention patterns across heads."""
    n_heads = attn_grid.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))

    if n_heads == 1:
        axes = [axes]

    for h in range(n_heads):
        sns.heatmap(attn_grid[h], ax=axes[h], cmap="viridis", cbar=True, vmin=0, vmax=1)
        axes[h].set_title(f"Head {h}")
        axes[h].set_xlabel("b")
        axes[h].set_ylabel("a")

    fig.suptitle(f"Attention (pos 1 -> pos 0) at Step {step} {title_suffix}", y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_attention_gif(checkpoint_dir: str, output_path: str, max_ckpts: int = 50):
    """Create an animated GIF of attention evolution from a directory of checkpoints."""
    # Find all checkpoints
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))

    # Extract steps and sort
    ckpts_with_steps = []
    for f in ckpt_files:
        try:
            step = int(f.split("checkpoint_")[1].split(".pt")[0])
            ckpts_with_steps.append((step, f))
        except ValueError:
            pass

    ckpts_with_steps.sort(key=lambda x: x[0])

    if not ckpts_with_steps:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    # Limit number of frames
    if len(ckpts_with_steps) > max_ckpts:
        indices = np.linspace(0, len(ckpts_with_steps) - 1, max_ckpts, dtype=int)
        ckpts_with_steps = [ckpts_with_steps[i] for i in indices]

    # Generate temporary images
    temp_dir = os.path.join(checkpoint_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    image_paths = []
    condition = os.path.basename(os.path.normpath(checkpoint_dir))

    print(f"Generating frames for {condition}...")
    for step, f in ckpts_with_steps:
        model, config, ckpt_step = load_checkpoint(f)
        prime = config.get('prime', 59)
        attn_grid = compute_attention_patterns(model, prime)

        img_path = os.path.join(temp_dir, f"frame_{step:06d}.png")
        generate_attention_heatmap(attn_grid, step, img_path, title_suffix=f"({condition})")
        image_paths.append(img_path)

    # Create GIF
    print(f"Creating GIF at {output_path}...")
    frames = []
    for img_path in image_paths:
        frames.append(imageio.v2.imread(img_path))

    imageio.mimsave(output_path, frames, fps=5)

    # Cleanup
    for img_path in image_paths:
        os.remove(img_path)
    os.rmdir(temp_dir)
    print(f"GIF saved to {output_path}")

def run_attention_analysis():
    """Run full attention analysis for available collapse levels."""
    base_dir = "results"
    output_dir = "results/analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    conditions = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

    for condition in conditions:
        cond_dir = os.path.join(base_dir, condition)
        if not os.path.exists(cond_dir):
            continue

        print(f"Processing {condition}...")

        # 1. Generate GIF
        gif_path = os.path.join(output_dir, f"{condition}_attention_evolution.gif")
        create_attention_gif(cond_dir, gif_path, max_ckpts=20)

        # 2. Extract key static checkpoints (e.g. pre, during, post grokking)
        # We can just pick specific steps
        ckpts = glob.glob(os.path.join(cond_dir, "checkpoint_*.pt"))
        if not ckpts:
            continue

        steps = sorted([int(f.split("checkpoint_")[1].split(".pt")[0]) for f in ckpts])

        if len(steps) > 3:
            key_steps = [steps[0], steps[len(steps)//2], steps[-1]]
        else:
            key_steps = steps

        for step in key_steps:
            ckpt_path = os.path.join(cond_dir, f"checkpoint_{step}.pt")
            model, config, _ = load_checkpoint(ckpt_path)
            attn_grid = compute_attention_patterns(model, config.get('prime', 59))

            out_path = os.path.join(output_dir, f"{condition}_attention_step_{step}.png")
            generate_attention_heatmap(attn_grid, step, out_path, title_suffix=f"({condition})")
            print(f"Saved {out_path}")

if __name__ == "__main__":
    run_attention_analysis()
