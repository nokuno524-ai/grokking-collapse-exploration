#!/usr/bin/env python3
"""
Attention Visualization Suite

Renders per-layer per-head attention matrices (heatmaps) and computes
token-position aggregate views (e.g., attention entropy per head) from
model checkpoints. Supports sweeping over a directory of checkpoints to
generate timelines of attention structure evolution.
"""

import argparse
import os
from pathlib import Path
import glob

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import ModularArithmeticTransformer
from src.attention_extraction import extract_attention, compute_entropy
from src.data import generate_modular_arithmetic, DatasetConfig
from src.checkpoint import load_run


def run_attention_extraction(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    """
    Run the model on inputs and extract attention weights.

    Args:
        model: Transformer model
        inputs: Input tensor (batch, seq_len)

    Returns:
        Attention weights array (layers, batch, heads, seq_len_q, seq_len_k)
    """
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with extract_attention(model) as attns:
        with torch.no_grad():
            _ = model(inputs)

    # Stack across layers
    # attns is a list of arrays (one per layer/call)
    # Return shape: (layers, batch, heads, seq_len, seq_len)
    return np.stack(attns)

def plot_attention_heatmaps(attn_weights: np.ndarray, title: str, output_path: str):
    """
    Plot attention heatmaps for each head in a single layer.

    Args:
        attn_weights: Attention weights for one layer (batch, heads, seq_len, seq_len)
                      We'll take the mean across the batch.
        title: Plot title
        output_path: Path to save the figure
    """
    # Average across batch
    mean_attn = np.mean(attn_weights, axis=0) # (heads, seq_len, seq_len)
    n_heads = mean_attn.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(mean_attn[i], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Head {i}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['a', 'b'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['a', 'b'])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Attention")
    parser.add_argument("--run-dir", type=str, help="Path to run directory")
    parser.add_argument("--step", type=int, default=None, help="Specific step to load")
    parser.add_argument("--output-dir", type=str, default="analysis/attention", help="Output directory")
    parser.add_argument("--prime", type=int, default=59, help="Prime modulus")
    parser.add_argument("--sweep", action="store_true", help="Sweep over all checkpoints in directory")

    args = parser.parse_args()

    if not args.run_dir:
        parser.print_help()
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate an evaluation batch (using the full domain for comprehensive attention)
    config = DatasetConfig(prime=args.prime)
    data = generate_modular_arithmetic(config)
    train_in, train_out, test_in, test_out = data
    eval_inputs = test_in # Evaluate on clean training points
    if len(eval_inputs) > 1024:
        # Subsample if too large
        idx = torch.randperm(len(eval_inputs))[:1024]
        eval_inputs = eval_inputs[idx]

    if args.sweep:
        # Find all checkpoints
        run_dir = Path(args.run_dir)
        ckpts = glob.glob(str(run_dir / "checkpoint_*.pt"))
        if not ckpts:
            print(f"No checkpoints found in {run_dir}")
            return

        def parse_step(p):
            try:
                return int(Path(p).stem.split("_")[1])
            except ValueError:
                return -1

        ckpts.sort(key=parse_step)
        steps = [parse_step(p) for p in ckpts if parse_step(p) >= 0]

        print(f"Found {len(steps)} checkpoints. Extracting attention...")

        all_entropies = []

        for step in tqdm(steps):
            model = load_run(args.run_dir, step=step)
            attns = run_attention_extraction(model, eval_inputs)

            # Save heatmaps for a few steps
            if step in [0, 100, 500, 1000, 5000, 10000, steps[-1]]:
                # Assuming 1 layer for now
                plot_attention_heatmaps(
                    attns[0],
                    f"Attention at step {step}",
                    os.path.join(args.output_dir, f"heatmap_step_{step}.png")
                )

            # Compute aggregate entropy (layer 0)
            entropy = compute_entropy(attns[0]) # (batch, heads, seq_len)
            mean_entropy = np.mean(entropy, axis=(0, 2)) # (heads,)
            all_entropies.append(mean_entropy)

        # Plot timeline
        all_entropies = np.array(all_entropies) # (n_steps, heads)
        plt.figure(figsize=(10, 6))
        for h in range(all_entropies.shape[1]):
            plt.plot(steps, all_entropies[:, h], label=f'Head {h}', marker='.', markersize=2)

        plt.xlabel('Training Step')
        plt.ylabel('Mean Attention Entropy')
        plt.title('Attention Entropy Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Log scale for x-axis if steps span multiple orders of magnitude
        if len(steps) > 1 and steps[-1] > 10 * steps[1]:
            plt.xscale('symlog', linthresh=100)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "entropy_timeline.png"), dpi=300)
        plt.close()
        print(f"Timeline saved to {args.output_dir}/entropy_timeline.png")

    else:
        model = load_run(args.run_dir, step=args.step)
        attns = run_attention_extraction(model, eval_inputs)

        # Save heatmap
        step_str = f"_step_{args.step}" if args.step is not None else "_final"
        out_path = os.path.join(args.output_dir, f"heatmap{step_str}.png")
        plot_attention_heatmaps(attns[0], f"Attention Heatmap", out_path)

        entropy = compute_entropy(attns[0])
        mean_entropy = np.mean(entropy, axis=(0, 2))
        print(f"Mean attention entropy per head: {mean_entropy}")
        print(f"Heatmap saved to {out_path}")

if __name__ == "__main__":
    main()
