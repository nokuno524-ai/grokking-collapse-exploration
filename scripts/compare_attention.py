import argparse
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.checkpoint import load_run
from src.attention_extraction import extract_attention
from src.data import generate_modular_arithmetic

def get_attention_stats(attn_matrix: torch.Tensor):
    """
    Compute summary stats for a single attention matrix of shape (seq_len, seq_len)
    """
    # 1. Row entropy
    # attn_matrix should already sum to 1 over last dim, but clamp for safety
    p = torch.clamp(attn_matrix, 1e-10, 1.0)
    entropy = -torch.sum(p * torch.log(p), dim=-1).mean().item()

    # 2. Diagonal dominance
    diag = torch.diagonal(attn_matrix).mean().item()

    # 3. Uniformity (how close to uniform attention it is)
    seq_len = attn_matrix.shape[-1]
    uniform_p = torch.ones_like(attn_matrix) / seq_len
    kl_div = torch.sum(p * torch.log(p / uniform_p), dim=-1).mean().item()

    return {
        "entropy": entropy,
        "diag_dominance": diag,
        "kl_uniform": kl_div
    }

def process_checkpoint(run_dir, step=None, batch_size=256, device="cpu"):
    """
    Process a single checkpoint, extracting attention patterns and stats.
    """
    model, step = load_run(run_dir, step=step)
    model.to(device)
    model.eval()

    # We need a batch of data
    # The prime should match the model
    prime = model.prime

    # Get a fixed batch of prompts
    torch.manual_seed(42)
    x, _ = generate_modular_arithmetic(prime, batch_size, 0)
    x = x.to(device)

    # Extract attention
    with torch.no_grad():
        with extract_attention(model) as attns:
            _ = model(x)

    # attns is a list (one per layer) of tensors (B, n_heads, seq_len, seq_len)
    # Average across batch to get per-head typical attention pattern
    avg_attns = []
    layer_head_stats = []

    for layer_idx, attn_tensor in enumerate(attns):
        # average over batch
        avg_attn = attn_tensor.mean(dim=0) # (n_heads, seq_len, seq_len)
        avg_attns.append(avg_attn.cpu().numpy())

        for head_idx in range(avg_attn.shape[0]):
            head_matrix = avg_attn[head_idx]
            stats = get_attention_stats(head_matrix)
            stats.update({
                "layer": layer_idx,
                "head": head_idx,
                "step": step,
            })
            layer_head_stats.append(stats)

    return avg_attns, layer_head_stats

def run_attention_analysis(run_dir: Path, out_dir: Path):
    """
    Analyze all checkpoints in a run.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = list(run_dir.glob("checkpoint_*.pt"))
    if not ckpts:
        print(f"No checkpoints found in {run_dir}")
        return

    steps = sorted([int(p.stem.split("_")[1]) for p in ckpts])

    all_stats = []

    for step in steps:
        print(f"Processing step {step}...")
        avg_attns, stats = process_checkpoint(run_dir, step)
        all_stats.extend(stats)

        # Plot attention heatmap for this step
        n_layers = len(avg_attns)
        n_heads = avg_attns[0].shape[0]

        fig, axes = plt.subplots(n_layers, n_heads, figsize=(3*n_heads, 3*n_layers))
        if n_layers == 1 and n_heads == 1:
            axes = np.array([[axes]])
        elif n_layers == 1:
            axes = axes[np.newaxis, :]
        elif n_heads == 1:
            axes = axes[:, np.newaxis]

        for l in range(n_layers):
            for h in range(n_heads):
                sns.heatmap(avg_attns[l][h], ax=axes[l, h], cmap="Blues", vmin=0, vmax=1)
                axes[l, h].set_title(f"L{l}H{h}")
                axes[l, h].set_xlabel("Key")
                axes[l, h].set_ylabel("Query")

        plt.tight_layout()
        plt.savefig(out_dir / f"attn_heatmap_step_{step}.png")
        plt.close(fig)

    # Save stats to CSV
    if all_stats:
        keys = all_stats[0].keys()
        with open(out_dir / "attention_stats.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_stats)

        # Plot trajectories
        import pandas as pd
        df = pd.DataFrame(all_stats)
        df['layer_head'] = df['layer'].astype(str) + "_" + df['head'].astype(str)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sns.lineplot(data=df, x="step", y="entropy", hue="layer_head", ax=axes[0])
        axes[0].set_title("Row Entropy")

        sns.lineplot(data=df, x="step", y="diag_dominance", hue="layer_head", ax=axes[1])
        axes[1].set_title("Diagonal Dominance")

        sns.lineplot(data=df, x="step", y="kl_uniform", hue="layer_head", ax=axes[2])
        axes[2].set_title("KL Divergence from Uniform")

        plt.tight_layout()
        plt.savefig(out_dir / "attention_trajectories.png")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--out-dir", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    run_attention_analysis(args.run_dir, args.out_dir)
