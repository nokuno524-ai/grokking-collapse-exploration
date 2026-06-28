import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Ensure imports work from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ModularArithmeticTransformer
from src.data import generate_modular_arithmetic, DatasetConfig

def get_checkpoints(run_dir):
    """Get all checkpoint paths sorted by step."""
    checkpoints = list(Path(run_dir).glob("checkpoint_*.pt"))
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))
    return checkpoints

def extract_attention(model, inputs):
    x = model.token_embed(inputs)
    x = x + model.pos_embed(torch.arange(inputs.shape[1], device=inputs.device))
    layer = model.transformer.layers[0]

    batch_first = getattr(layer.self_attn, 'batch_first', False)
    if not batch_first:
        x = x.transpose(0, 1)

    attn_output, attn_weights = layer.self_attn(
        x, x, x,
        need_weights=True,
        average_attn_weights=False
    )
    return attn_weights.detach().cpu().numpy()

def compute_entropy(attn_weights):
    eps = 1e-10
    entropy = -np.sum(attn_weights * np.log2(attn_weights + eps), axis=-1)
    return np.mean(entropy, axis=(0, 2))

def plot_attention_evolution(run_dir, output_path, condition_name="Pure"):
    checkpoints = get_checkpoints(run_dir)
    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints. Processing...")

    import json
    with open(Path(run_dir) / 'results.json') as f:
        config = json.load(f)['config']

    model = ModularArithmeticTransformer(
        prime=config['prime'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_layers=config['n_layers']
    )

    data_config = DatasetConfig(prime=config['prime'], train_fraction=1.0)
    inputs, _, _, _ = generate_modular_arithmetic(data_config)
    # Take a small batch
    inputs = inputs[:1] # Just use 1 sample for heatmaps

    steps = []
    entropies = []
    head_norms = []

    # Setup for heatmaps
    num_ckpts = len(checkpoints)
    num_heads = config['n_heads']
    fig_heat, axes_heat = plt.subplots(num_heads, num_ckpts, figsize=(num_ckpts * 2, num_heads * 2))

    for i, ckpt_path in enumerate(checkpoints):
        step = int(ckpt_path.stem.split('_')[1])
        steps.append(step)

        try:
            ckpt = torch.load(ckpt_path, weights_only=True, map_location='cpu')
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            else:
                model.load_state_dict(ckpt)
        except Exception:
            ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            else:
                model.load_state_dict(ckpt)

        model.eval()

        attn_weights = extract_attention(model, inputs)
        entropy = compute_entropy(attn_weights)
        entropies.append(entropy)
        norm = np.std(attn_weights, axis=(0, 2, 3))
        head_norms.append(norm)

        # Draw heatmaps
        # attn_weights shape: (1, heads, 2, 2) since we use batch=1, seq=2 for modular arithmetic
        sample_attn = attn_weights[0]
        for h in range(num_heads):
            ax = axes_heat[h, i] if num_ckpts > 1 else axes_heat[h]
            sns.heatmap(sample_attn[h], ax=ax, cmap="YlGnBu", cbar=False, vmin=0, vmax=1)
            if h == 0:
                ax.set_title(f"Step {step}")
            if i == 0:
                ax.set_ylabel(f"Head {h}")
            ax.set_xticks([])
            ax.set_yticks([])

    fig_heat.suptitle(f"Attention Heatmap Evolution ({condition_name})", fontsize=16)
    plt.tight_layout()
    heatmap_out = output_path.replace('.png', '_heatmaps.png')
    fig_heat.savefig(heatmap_out, dpi=300, bbox_inches='tight')
    plt.close(fig_heat)

    entropies = np.array(entropies)
    head_norms = np.array(head_norms)

    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for head in range(config['n_heads']):
        ax1.plot(steps, entropies[:, head], label=f'Head {head}', lw=2)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Attention Entropy (bits)', fontsize=12)
    ax1.set_title(f'Attention Entropy Evolution ({condition_name})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for head in range(config['n_heads']):
        ax2.plot(steps, head_norms[:, head], label=f'Head {head}', lw=2)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Attention Std Dev', fontsize=12)
    ax2.set_title(f'Attention Head Role Evolution ({condition_name})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved attention evolution plots to {output_path} and {heatmap_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pure-dir", type=str, default="results/pure", help="Path to pure run")
    parser.add_argument("--collapse-dir", type=str, default="results/medium_collapse", help="Path to collapsed run")
    parser.add_argument("--out-dir", type=str, default="plots", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.pure_dir):
        plot_attention_evolution(args.pure_dir, os.path.join(args.out_dir, "attention_evolution_pure.png"), "Pure")

    if os.path.exists(args.collapse_dir):
        plot_attention_evolution(args.collapse_dir, os.path.join(args.out_dir, "attention_evolution_collapsed.png"), "Collapsed")
