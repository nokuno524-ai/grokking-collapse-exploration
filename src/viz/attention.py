import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import json

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.model import ModularArithmeticTransformer

def extract_attention_weights(model: nn.Module, dummy_input: torch.Tensor) -> torch.Tensor:
    """
    Extract multi-head attention weights from ModularArithmeticTransformer.
    Since PyTorch's TransformerEncoderLayer hardcodes need_weights=False,
    we have to manually reconstruct them from the projections.
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings
        tok = model.token_embed(dummy_input)
        batch_size = dummy_input.shape[0]
        positions = torch.arange(2, device=dummy_input.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)
        x = tok + pos

        # Manually extract Q, K, V from the first layer
        attn_layer = model.transformer.layers[0].self_attn
        d_model = model.d_model
        n_heads = model.n_heads
        head_dim = d_model // n_heads

        # in_proj_weight is shape (3 * d_model, d_model)
        # in_proj_bias is shape (3 * d_model)
        w_q, w_k, w_v = attn_layer.in_proj_weight.chunk(3)
        b_q, b_k, b_v = attn_layer.in_proj_bias.chunk(3)

        q = (x @ w_q.T + b_q).view(batch_size, 2, n_heads, head_dim).transpose(1, 2)
        k = (x @ w_k.T + b_k).view(batch_size, 2, n_heads, head_dim).transpose(1, 2)

        # Scaled dot product
        scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        return attn_weights  # (batch, n_heads, 2, 2)

def plot_attention_grid(attn_weights: torch.Tensor, step: int, output_dir: Path):
    """
    Plot a grid of attention maps for each head.
    """
    # Average across batch
    avg_attn = attn_weights.mean(dim=0).cpu().numpy()  # (n_heads, 2, 2)
    n_heads = avg_attn.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(3 * n_heads, 3))
    if n_heads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(avg_attn[i], vmin=0, vmax=1, cmap='viridis')
        ax.set_title(f'Head {i}')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['a', 'b'])
        ax.set_yticklabels(['a', 'b'])

    plt.suptitle(f'Attention Patterns at Step {step}')
    plt.tight_layout()
    plt.savefig(output_dir / f'attn_grid_step_{step}.png')
    plt.close()

def plot_head_specialization(attn_history: List[torch.Tensor], steps: List[int], output_dir: Path):
    """
    Cluster head specialization using PCA on their flattened attention matrices across steps.
    """
    if not HAS_SKLEARN:
        print("Warning: scikit-learn is not installed. Skipping head specialization PCA plot.")
        return

    # attn_history is list of (batch, n_heads, 2, 2)
    # We want to track each head's evolution.

    n_steps = len(steps)
    n_heads = attn_history[0].shape[1]

    # Flatten attention weights for PCA: (n_heads * n_steps, 4)
    data = []
    labels = []

    for step_idx, attn in enumerate(attn_history):
        avg_attn = attn.mean(dim=0).cpu().numpy() # (n_heads, 2, 2)
        for h in range(n_heads):
            data.append(avg_attn[h].flatten())
            labels.append(h)

    data = np.array(data)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    for h in range(n_heads):
        idx = [i for i, label in enumerate(labels) if label == h]
        plt.plot(reduced[idx, 0], reduced[idx, 1], marker='o', label=f'Head {h}', alpha=0.7)
        # Mark start and end
        plt.scatter(reduced[idx[0], 0], reduced[idx[0], 1], color='black', marker='^', s=100)
        plt.scatter(reduced[idx[-1], 0], reduced[idx[-1], 1], color='black', marker='*', s=150)

    plt.title('PCA of Head Attention Evolution')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'head_specialization_pca.png')
    plt.close()

def plot_timeline_overlay(history: List[Dict[str, Any]], output_dir: Path):
    """
    Plot weight norm and test accuracy on the same timeline.
    """
    steps = [h['step'] for h in history]
    acc = [h.get('test_acc', 0) for h in history]
    norm = [h.get('weight_norm', 0) for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Test Accuracy', color=color)
    ax1.plot(steps, acc, color=color, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Weight Norm', color=color)
    ax2.plot(steps, norm, color=color, linewidth=2, linestyle='--', label='Weight Norm')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Grokking vs Weight Norm Dynamics')
    fig.tight_layout()
    plt.savefig(output_dir / 'timeline_overlay.png')
    plt.close()

def demo():
    """
    Run synthetic data demo to generate outputs without real checkpoints.
    """
    print("Running Attention Visualizer Demo...")
    output_dir = Path("results/viz_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Dummy History
    print("Generating dummy history...")
    history = []
    for step in range(0, 2000, 100):
        acc = 0.1 if step < 1000 else min(1.0, 0.1 + (step - 1000) * 0.002)
        norm = 100.0 if step < 500 else max(20.0, 100.0 - (step - 500) * 0.05)
        history.append({"step": step, "test_acc": acc, "weight_norm": norm})

    plot_timeline_overlay(history, output_dir)

    # 2. Generate Dummy Checkpoints and Extract Attention
    print("Generating dummy checkpoints and attention patterns...")
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=4, d_ff=64)
    dummy_input = torch.randint(0, 59, (16, 2))

    attn_history = []
    steps = [0, 500, 1000, 1500]

    for step in steps:
        # Simulate training by adding some noise to weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        attn = extract_attention_weights(model, dummy_input)
        attn_history.append(attn)
        plot_attention_grid(attn, step, output_dir)

    # 3. Plot Head Specialization
    print("Clustering head specialization...")
    plot_head_specialization(attn_history, steps, output_dir)

    print(f"Demo complete! Outputs saved to {output_dir}")

if __name__ == "__main__":
    demo()
