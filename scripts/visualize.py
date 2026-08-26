"""
Visualization script for attention entropy evolution,
loss/accuracy curves, and weight-norm scatter plots.
"""
import json
import os
import glob
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy

# For synthetic checkpoints if needed
from src.model import ModularArithmeticTransformer
from src.train import TrainConfig

def compute_attention_entropy(attn_weights):
    """
    Compute entropy of attention weights for each head.
    attn_weights: (batch, n_heads, seq_len, seq_len)
    Returns: (n_heads,) mean entropy
    """
    # Average over batch and sequence dimension
    # Actually, we want entropy over the key distribution for each query
    # So we compute entropy for each row, then average
    eps = 1e-10
    # Flatten across batch and queries
    # attn_weights shape is typically (batch, n_heads, query_len, key_len)
    n_heads = attn_weights.shape[1]

    # Entropy over the key_len dimension
    entropies = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)

    # Mean over batch and query_len
    return entropies.mean(dim=(0, 2)).detach().numpy()

def compute_head_specialization(attn_weights):
    """
    Compute head specialization as the entropy of the *average* attention distribution.
    A specialized head will have low entropy of its average attention.
    """
    eps = 1e-10
    # Average across batch and queries
    avg_attn = attn_weights.mean(dim=(0, 2))  # (n_heads, key_len)

    # Entropy of this average distribution
    specialization = -torch.sum(avg_attn * torch.log(avg_attn + eps), dim=-1)
    return specialization.detach().numpy()

def plot_attention_evolution(checkpoint_dir=None, model_config=None, output_path="figures/attention_evolution.png"):
    if checkpoint_dir is None:
        checkpoint_dir = Path("results/pure")
        if not checkpoint_dir.exists():
            print("results/pure not found, skipping attention evolution.")
            return
    else:
        checkpoint_dir = Path(checkpoint_dir)

    grokking_step = 1400
    config = model_config or {}
    res_path = checkpoint_dir / "results.json"
    if res_path.exists():
        with open(res_path) as f:
            res = json.load(f)
        grokking_step = res.get("grokking_step", 1400)
        config = res.get("config", config)

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    steps = []
    entropies = []
    specializations = []

    # We need a dummy input to compute attention
    device = torch.device("cpu")
    dummy_input = torch.randint(0, config.get("prime", 59), (32, 2))

    model = ModularArithmeticTransformer(
        prime=config.get("prime", 59),
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        d_ff=config.get("d_ff", 512),
        n_layers=config.get("n_layers", 1)
    )

    # We need to monkey patch self_attn to capture weights
    captured_weights = []
    def hook(module, input, output):
        # output is (attn_out, attn_weights) if need_weights=True
        # However, nn.MultiheadAttention returns weights as 2nd element
        # But wait, nn.TransformerEncoderLayer defaults to need_weights=False inside
        pass

    # Better approach: directly compute QK scaled dot product
    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.split("_")[1])
        steps.append(step)

        # Load state dict
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state", ckpt)
        # Handle "module." prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        with torch.no_grad():
            # Manually compute attention for the first layer
            # 1. Get embeddings
            tok = model.token_embed(dummy_input)
            pos = model.pos_embed(torch.arange(2).unsqueeze(0).expand(32, -1))
            h = tok + pos

            # 2. Get Q, K from self_attn in encoder_layer
            layer = model.transformer.layers[0]
            # PyTorch's MultiheadAttention uses in_proj_weight
            in_proj_weight = layer.self_attn.in_proj_weight
            in_proj_bias = layer.self_attn.in_proj_bias

            qkv = torch.nn.functional.linear(h, in_proj_weight, in_proj_bias)
            d_model = model.d_model
            q, k, v = qkv.split(d_model, dim=-1)

            # Reshape for heads
            n_heads = model.n_heads
            head_dim = d_model // n_heads
            batch_size = h.size(0)

            q = q.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

            # Scaled dot product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)

            ent = compute_attention_entropy(attn_weights)
            spec = compute_head_specialization(attn_weights)

            entropies.append(ent)
            specializations.append(spec)

    entropies = np.array(entropies)
    specializations = np.array(specializations)

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for h in range(entropies.shape[1]):
        plt.plot(steps, entropies[:, h], label=f'Head {h+1}')
    plt.axvline(x=grokking_step, color='r', linestyle='--', label='Grokking Step')
    plt.xlabel('Training Steps')
    plt.ylabel('Attention Entropy')
    plt.title('Per-Head Attention Entropy')
    plt.legend()

    plt.subplot(1, 2, 2)
    for h in range(specializations.shape[1]):
        plt.plot(steps, specializations[:, h], label=f'Head {h+1}')
    plt.axvline(x=grokking_step, color='r', linestyle='--', label='Grokking Step')
    plt.xlabel('Training Steps')
    plt.ylabel('Head Specialization (Entropy of Avg Attn)')
    plt.title('Head Specialization')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

    main()

def plot_curves_and_scatter():
    if not Path("results/summary.csv").exists():
        print("results/summary.csv not found. Please run scripts/inventory.py first.")
        return
    summary_df = pd.read_csv("results/summary.csv")

    # 1. Weight-norm vs grokking-step scatter
    plt.figure(figsize=(8, 6))
    grokked = summary_df.dropna(subset=['grokking_step'])
    not_grokked = summary_df[summary_df['grokking_step'].isna()]

    # We will use collapse_ratio for colors
    scatter = plt.scatter(grokked['grokking_step'], grokked['final_weight_norm'],
                c=grokked['collapse_ratio'], cmap='viridis', s=100, alpha=0.7)

    # Plot not_grokked at right edge
    max_step = grokked['grokking_step'].max() if not grokked.empty else 50000
    if not not_grokked.empty:
        plt.scatter([max_step * 1.1] * len(not_grokked), not_grokked['final_weight_norm'],
                    c=not_grokked['collapse_ratio'], cmap='viridis', marker='X', s=100, alpha=0.7)
        plt.axvline(x=max_step * 1.05, color='gray', linestyle='--')
        plt.text(max_step * 1.1, plt.ylim()[0], "Failed to Grok", rotation=90, va='bottom')

    plt.colorbar(scatter, label='Collapse Ratio')
    plt.xlabel('Grokking Step')
    plt.ylabel('Final Weight Norm')
    plt.title('Weight Norm vs Grokking Step')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/weight_norm_scatter.png")
    plt.close()
    print("Saved figures/weight_norm_scatter.png")

    # 2. Loss/accuracy curves faceted by collapse level
    # We need history for this, which means re-reading results.json for a representative run per condition
    plt.figure(figsize=(15, 5))

    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    for cond in conditions:
        path = Path(f"results/{cond}/results.json")
        if path.exists():
            with open(path) as f:
                res = json.load(f)

            history = res.get("history", [])
            if not history:
                continue

            steps = [h["step"] for h in history]
            train_acc = [h["train_acc"] for h in history]
            test_acc = [h["test_acc"] for h in history]

            # Subplot 1: Train Acc
            plt.subplot(1, 2, 1)
            plt.plot(steps, train_acc, label=f"{cond} Train")

            # Subplot 2: Test Acc
            plt.subplot(1, 2, 2)
            plt.plot(steps, test_acc, label=f"{cond} Test")

    plt.subplot(1, 2, 1)
    plt.xlabel("Steps")
    plt.ylabel("Train Accuracy")
    plt.title("Train Accuracy by Condition")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.xlabel("Steps")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy by Condition")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/accuracy_curves.png")
    plt.close()
    print("Saved figures/accuracy_curves.png")


def main():
    Path("figures").mkdir(exist_ok=True)
    plot_attention_evolution()
    plot_curves_and_scatter()

if __name__ == "__main__":
    main()
