"""
Attention pattern extraction and visualization utilities.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.model import ModularArithmeticTransformer


def load_model_from_checkpoint(ckpt_path: Path) -> ModularArithmeticTransformer:
    """Load model weights and config from a checkpoint file."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})

    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1)
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def extract_attention_weights(model: ModularArithmeticTransformer, x: torch.Tensor) -> torch.Tensor:
    """
    Extract attention weights from the first transformer layer.

    Args:
        model: The trained ModularArithmeticTransformer.
        x: Input tensor of shape (batch, 2).

    Returns:
        Tensor of shape (batch, n_heads, seq_len, seq_len) containing attention weights.
    """
    with torch.no_grad():
        # Input embeddings + positional embeddings
        tok = model.token_embed(x)
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        pos = model.pos_embed(positions)
        h = tok + pos

        # Manually perform the first layer forward pass to get attention weights
        layer = model.transformer.layers[0]

        # Apply norm1 if standard PyTorch TransformerEncoderLayer
        src = h
        if hasattr(layer, 'norm1'):
            src_norm = layer.norm1(src)
        else:
            src_norm = src

        # self_attn expects query, key, value
        attn_out, attn_weights = layer.self_attn(
            src_norm, src_norm, src_norm,
            need_weights=True,
            average_attn_weights=False
        )

        return attn_weights


def plot_attention_heatmap(attn_weights: torch.Tensor, head_idx: int, output_path: Path):
    """Plot average attention heatmap for a specific head over the batch."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    avg_attn = attn_weights[:, head_idx, :, :].mean(dim=0).detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(avg_attn, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Attention Weight")
    plt.title(f"Head {head_idx} Attention")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.xticks([0, 1], ["pos 0", "pos 1"])
    plt.yticks([0, 1], ["pos 0", "pos 1"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_evolution(condition_dir: Path, output_dir: Path, prime: int = 59):
    """Plot how attention patterns evolve over training for a specific condition."""
    if not HAS_MATPLOTLIB:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(condition_dir.glob("checkpoint_*.pt"), key=lambda x: int(x.stem.split("_")[1]))

    if not checkpoints:
        print(f"No checkpoints found in {condition_dir}")
        return

    # Generate evaluation data
    torch.manual_seed(42)
    x = torch.randint(0, prime, (100, 2))

    model = load_model_from_checkpoint(checkpoints[0])
    num_heads = model.n_heads

    for head_idx in range(num_heads):
        steps = []
        attn_to_0 = []
        attn_to_1 = []

        for ckpt_path in checkpoints:
            try:
                step = int(ckpt_path.stem.split("_")[1])
                model = load_model_from_checkpoint(ckpt_path)
                attn = extract_attention_weights(model, x)

                # Average attention over queries and batch
                avg_attn = attn[:, head_idx, :, :].mean(dim=0).detach().cpu().numpy()

                steps.append(step)
                attn_to_0.append(avg_attn[:, 0].mean())
                attn_to_1.append(avg_attn[:, 1].mean())
            except Exception as e:
                print(f"Error processing {ckpt_path}: {e}")

        plt.figure(figsize=(10, 5))
        plt.plot(steps, attn_to_0, label="To pos 0", linewidth=2)
        plt.plot(steps, attn_to_1, label="To pos 1", linewidth=2)
        plt.xlabel("Training Step")
        plt.ylabel("Average Attention Weight")
        plt.title(f"{condition_dir.name} - Head {head_idx} Attention Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)

        plt.savefig(output_dir / f"{condition_dir.name}_head_{head_idx}_evolution.png", dpi=150, bbox_inches='tight')
        plt.close()


def compare_collapse_attention(results_dir: Path, output_path: Path, step: int = 50000, prime: int = 59):
    """Create side-by-side comparison plots of attention patterns across collapse levels."""
    if not HAS_MATPLOTLIB:
        return

    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    valid_conditions = []
    condition_attns = {}

    torch.manual_seed(42)
    x = torch.randint(0, prime, (200, 2))

    num_heads = 4
    for condition in conditions:
        ckpt_path = results_dir / condition / f"checkpoint_{step}.pt"
        if ckpt_path.exists():
            try:
                model = load_model_from_checkpoint(ckpt_path)
                num_heads = model.n_heads
                attn = extract_attention_weights(model, x)
                # Average over batch
                condition_attns[condition] = attn.mean(dim=0).detach().cpu().numpy()
                valid_conditions.append(condition)
            except Exception as e:
                print(f"Error loading {ckpt_path}: {e}")

    if not valid_conditions:
        print(f"No valid checkpoints found for step {step}")
        return

    fig, axes = plt.subplots(num_heads, len(valid_conditions), figsize=(4*len(valid_conditions), 3*num_heads))

    # Handle single row or single column cases
    if num_heads == 1 and len(valid_conditions) == 1:
        axes = np.array([[axes]])
    elif num_heads == 1:
        axes = axes[np.newaxis, :]
    elif len(valid_conditions) == 1:
        axes = axes[:, np.newaxis]

    for i, condition in enumerate(valid_conditions):
        attn_matrix = condition_attns[condition]
        for h in range(num_heads):
            ax = axes[h, i]
            im = ax.imshow(attn_matrix[h], cmap="viridis", vmin=0, vmax=1)

            if h == 0:
                ax.set_title(condition.replace("_", " ").title())
            if i == 0:
                ax.set_ylabel(f"Head {h}\nQuery")
            else:
                ax.set_yticks([])

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["pos 0", "pos 1"])
            if i == 0:
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["pos 0", "pos 1"])

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Attention Weight", fraction=0.02, pad=0.04)
    plt.suptitle(f"Attention Patterns Across Collapse Conditions (Step {step})", y=1.02, fontsize=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
