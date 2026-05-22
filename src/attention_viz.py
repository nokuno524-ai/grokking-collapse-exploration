import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.model import ModularArithmeticTransformer

def extract_attention_weights(model: ModularArithmeticTransformer, x: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract attention weights from all transformer layers.

    Args:
        model: ModularArithmeticTransformer instance
        x: Input tensor of shape (batch, 2)

    Returns:
        List of attention_weights tensors, each of shape (batch, n_heads, seq_len, seq_len)
    """
    model.eval()
    with torch.no_grad():
        batch_size = x.shape[0]
        tok = model.token_embed(x)
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)
        h = tok + pos

        attn_weights_all_layers = []
        # In ModularArithmeticTransformer, transformer is nn.TransformerEncoder
        # which has a ModuleList called 'layers'
        for layer in model.transformer.layers:
            # PyTorch TransformerEncoderLayer self_attn returns (attn_output, attn_weights)
            _, attn_weights = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
            attn_weights_all_layers.append(attn_weights)
            h = layer(h) # Need to process through layer to get correct input for next layer

        return attn_weights_all_layers

def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention distributions.

    Args:
        attention_weights: shape (batch, n_heads, seq_len, seq_len)

    Returns:
        entropy: shape (batch, n_heads, seq_len)
    """
    # attention_weights sum to 1 over the last dimension
    # Add epsilon to avoid log(0)
    eps = 1e-10
    entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=-1)
    return entropy

def compute_attention_entropy_all_layers(attention_weights_list: List[torch.Tensor]) -> List[torch.Tensor]:
    return [compute_attention_entropy(aw) for aw in attention_weights_list]

def plot_attention_heatmaps(attention_weights: torch.Tensor, tokens: List[str], title: str, output_path: Path):
    """
    Plot attention heatmaps for all heads for a single example.

    Args:
        attention_weights: shape (n_heads, seq_len, seq_len)
        tokens: list of token strings for the axes
        title: plot title
        output_path: where to save
    """
    n_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for i in range(n_heads):
        sns.heatmap(
            attention_weights[i].cpu().numpy(),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=tokens,
            yticklabels=tokens,
            ax=axes[i],
            cbar=(i == n_heads - 1)
        )
        axes[i].set_title(f"Head {i+1}")
        axes[i].set_xlabel("Key / Value")
        if i == 0:
            axes[i].set_ylabel("Query")

    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_evolution(results_dir: Path, output_path: Path):
    """
    Plot how attention entropy evolves over training.
    """
    checkpoints = sorted([p for p in results_dir.glob("checkpoint_*.pt")], key=lambda x: int(x.stem.split('_')[1]))
    if not checkpoints:
        print(f"No checkpoints found in {results_dir}")
        return

    # Load config to get model params
    config_path = results_dir / "results.json"
    if not config_path.exists():
        print(f"No config found in {results_dir}")
        return

    with open(config_path) as f:
        data = json.load(f)
        config = data.get("config", {})

    prime = config.get("prime", 59)
    d_model = config.get("d_model", 128)
    n_heads = config.get("n_heads", 4)

    steps = []
    avg_entropies = []

    # Test on a few equations
    x = torch.tensor([[10, 20], [0, 5], [58, 58]])

    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.split('_')[1])
        steps.append(step)

        model = ModularArithmeticTransformer(prime=prime, d_model=d_model, n_heads=n_heads)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model_state" in state_dict:
            model.load_state_dict(state_dict["model_state"])
        else:
            model.load_state_dict(state_dict)

        attn = extract_attention_weights(model, x)
        attn_l0 = attn[0] # (batch, n_heads, seq_len, seq_len)
        entropy = compute_attention_entropy(attn_l0)  # (batch, n_heads, seq_len)
        # Average over batch, heads, and sequence length
        avg_entropies.append(entropy.mean().item())

        # Save heatmap for the first equation at this step
        if step in [5000, 25000, 50000]:
            heatmap_path = results_dir / f"attn_heatmap_step_{step}.png"
            tokens = [f"Pos 0 (a={x[0,0].item()})", f"Pos 1 (b={x[0,1].item()})"]
            plot_attention_heatmaps(attn_l0[0], tokens, f"Attention at Step {step}", heatmap_path)

    # Plot entropy evolution
    plt.figure(figsize=(8, 5))
    plt.plot(steps, avg_entropies, marker='o', linewidth=2)
    plt.title("Attention Entropy Evolution")
    plt.xlabel("Training Step")
    plt.ylabel("Average Attention Entropy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def compare_attention_across_conditions(base_results_dir: Path, output_path: Path):
    """
    Compare final attention entropy across different conditions.
    """
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    condition_names = []
    final_entropies = []

    x = torch.tensor([[10, 20], [0, 5], [58, 58]])

    for condition in conditions:
        cond_dir = base_results_dir / condition
        if not cond_dir.exists():
            continue

        # Find final checkpoint
        checkpoints = sorted([p for p in cond_dir.glob("checkpoint_*.pt")], key=lambda x: int(x.stem.split('_')[1]))
        if not checkpoints:
            continue

        ckpt_path = checkpoints[-1]

        config_path = cond_dir / "results.json"
        if not config_path.exists():
            continue

        with open(config_path) as f:
            data = json.load(f)
            config = data.get("config", {})

        prime = config.get("prime", 59)
        d_model = config.get("d_model", 128)
        n_heads = config.get("n_heads", 4)

        model = ModularArithmeticTransformer(prime=prime, d_model=d_model, n_heads=n_heads)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model_state" in state_dict:
            model.load_state_dict(state_dict["model_state"])
        else:
            model.load_state_dict(state_dict)

        attn = extract_attention_weights(model, x)
        entropy = compute_attention_entropy(attn[0])

        condition_names.append(condition.replace("_", "\n"))
        final_entropies.append(entropy.mean().item())

    if not condition_names:
        print("No valid conditions found for comparison.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=condition_names, y=final_entropies, palette="viridis", hue=condition_names, legend=False)
    plt.title("Final Attention Entropy by Condition")
    plt.xlabel("Condition")
    plt.ylabel("Attention Entropy (lower = sharper)")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Test script locally
    model = ModularArithmeticTransformer()
    x = torch.tensor([[10, 20]])
    attn = extract_attention_weights(model, x)
    entropy = compute_attention_entropy(attn[0])
    print("Attention extraction layers:", len(attn))
    print("Entropy shape:", entropy.shape)

    # Process results if run as main
    results_dir = Path("results")
    if results_dir.exists():
        pure_dir = results_dir / "pure"
        if pure_dir.exists():
            plot_attention_evolution(pure_dir, results_dir / "attn_evolution_pure.png")
            print("Generated attention evolution for 'pure' condition.")

        compare_attention_across_conditions(results_dir, results_dir / "attn_comparison.png")
        print("Generated cross-condition comparison.")
