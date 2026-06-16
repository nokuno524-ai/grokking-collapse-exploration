"""
Analysis of attention pattern evolution over the course of training.
Tracks how attention head focus and entropy change, helping to identify
mechanistic phase transitions that align with grokking.
"""

import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from src.model import ModularArithmeticTransformer


def extract_attention_patterns(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Extract attention patterns from the model's transformer layer for a given input batch.

    Args:
        model: ModularArithmeticTransformer model
        inputs: Input tensor of shape (batch, 2)

    Returns:
        torch.Tensor: Attention weights of shape (batch, n_heads, seq_len, seq_len)
    """
    # The ModularArithmeticTransformer uses nn.TransformerEncoderLayer
    # By default, PyTorch's MultiheadAttention doesn't easily return per-head weights
    # unless we hook or pass need_weights=True.
    # Fortunately, the prompt hints: "Attention weights ... can be extracted directly
    # without manual Q/K/V computations by calling the self_attn method of the
    # nn.TransformerEncoderLayer with need_weights=True and average_attn_weights=False."

    batch_size = inputs.shape[0]
    device = inputs.device

    # 1. Forward through embeddings
    tok = model.token_embed(inputs)  # (batch, seq_len, d_model)
    seq_len = inputs.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)  # (batch, seq_len, d_model)
    h = tok + pos  # (batch, seq_len, d_model)

    layer = model.transformer.layers[0]

    # batch_first is True in this model's TransformerEncoderLayer
    # PyTorch self_attn expects query, key, value
    attn_output, attn_weights = layer.self_attn(
        query=h,
        key=h,
        value=h,
        need_weights=True,
        average_attn_weights=False
    )

    # attn_weights shape should be (batch, n_heads, seq_len, seq_len)
    return attn_weights.detach().cpu()


def compute_attention_entropy(patterns: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of attention weights per head.
    High entropy = diffuse attention, Low entropy = focused attention.

    Args:
        patterns: Attention weights of shape (batch, n_heads, seq_len, seq_len)

    Returns:
        torch.Tensor: Average entropy per head of shape (n_heads,)
    """
    # patterns are probability distributions over the last dimension (seq_len)
    # Entropy = -sum(p * log(p))
    # Handle zeros
    p = patterns + 1e-10
    entropy = -(p * torch.log(p)).sum(dim=-1)  # (batch, n_heads, seq_len)

    # Average over batch and seq_len
    avg_entropy = entropy.mean(dim=(0, 2))  # (n_heads,)
    return avg_entropy


def track_attention_evolution(
    run_dir: str,
    inputs: torch.Tensor,
    model_config: dict
) -> Dict[int, torch.Tensor]:
    """
    Load all checkpoints from a run chronologically, and extract attention
    patterns and their entropies.

    Args:
        run_dir: Directory containing checkpoint_*.pt files.
        inputs: Standard batch of inputs to evaluate attention on.
        model_config: Dict of model hyperparams to instantiate the model.

    Returns:
        Dict mapping step_number to a dictionary of metrics.
    """
    pattern = os.path.join(run_dir, "checkpoint_*.pt")
    checkpoints = glob.glob(pattern)

    # Sort by step number
    def get_step(p):
        name = os.path.basename(p)
        return int(name.replace("checkpoint_", "").replace(".pt", ""))

    checkpoints.sort(key=get_step)

    model = ModularArithmeticTransformer(**model_config)
    model.eval()

    evolution = {}

    for ckpt_path in checkpoints:
        step = get_step(ckpt_path)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        sd = ckpt.get("model_state", ckpt)
        model.load_state_dict(sd)

        patterns = extract_attention_patterns(model, inputs)
        entropy = compute_attention_entropy(patterns)

        evolution[step] = {
            "patterns": patterns.mean(dim=0),  # Average over batch for storage: (n_heads, seq_len, seq_len)
            "entropy": entropy
        }

    return evolution


def detect_attention_phase_transitions(evolution: Dict[int, Dict]) -> List[int]:
    """
    Identify sudden shifts in attention entropy.

    Args:
        evolution: The output of track_attention_evolution.

    Returns:
        List of step numbers where significant entropy drops/spikes occur.
    """
    steps = sorted(list(evolution.keys()))
    if len(steps) < 2:
        return []

    # We'll compute the discrete derivative of mean entropy
    entropies = [evolution[s]["entropy"].mean().item() for s in steps]
    diffs = np.diff(entropies)

    # Threshold for a 'sudden shift'
    threshold = np.std(diffs) * 2

    transitions = []
    for i, diff in enumerate(diffs):
        if abs(diff) > threshold:
            transitions.append(steps[i+1])

    return transitions


def plot_attention_evolution(evolution: Dict[int, Dict], output_dir: str):
    """
    Generate publication-quality figures: attention heatmap grids,
    entropy evolution line plots, and head importance rankings.

    Args:
        evolution: Dict output from track_attention_evolution.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    steps = sorted(list(evolution.keys()))
    if not steps:
        return

    # 1. Entropy Evolution Line Plot
    n_heads = evolution[steps[0]]["entropy"].shape[0]

    plt.figure(figsize=(10, 6))
    for h in range(n_heads):
        h_entropies = [evolution[s]["entropy"][h].item() for s in steps]
        plt.plot(steps, h_entropies, label=f"Head {h}")

    # Plot mean entropy as a thick line
    mean_entropies = [evolution[s]["entropy"].mean().item() for s in steps]
    plt.plot(steps, mean_entropies, color='black', linewidth=3, linestyle='--', label="Mean Entropy")

    plt.title("Attention Entropy Evolution")
    plt.xlabel("Step")
    plt.ylabel("Shannon Entropy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "attention_entropy_evolution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Attention Heatmaps at Start, Middle, End
    if len(steps) >= 3:
        key_steps = [steps[0], steps[len(steps)//2], steps[-1]]
    else:
        key_steps = steps

    for step in key_steps:
        patterns = evolution[step]["patterns"] # (n_heads, 2, 2)
        fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
        if n_heads == 1:
            axes = [axes]

        for h in range(n_heads):
            sns.heatmap(patterns[h].numpy(), ax=axes[h], vmin=0, vmax=1, cmap="YlGnBu", annot=True)
            axes[h].set_title(f"Head {h}")
            axes[h].set_xlabel("Key Position")
            axes[h].set_ylabel("Query Position")

        plt.suptitle(f"Attention Patterns at Step {step}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"attention_heatmaps_step_{step}.png"), dpi=300, bbox_inches="tight")
        plt.close()
