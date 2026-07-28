import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ModularArithmeticTransformer

def extract_attention_weights(model: ModularArithmeticTransformer, x: torch.Tensor) -> torch.Tensor:
    """
    Manually extract true attention weights from ModularArithmeticTransformer since
    nn.TransformerEncoderLayer uses need_weights=False by default in recent PyTorch versions.

    Args:
        model: ModularArithmeticTransformer instance
        x: Input tensor of shape (batch, 2)

    Returns:
        Attention weights of shape (batch, n_heads, seq_len, seq_len)
    """
    batch_size = x.shape[0]

    with torch.no_grad():
        # Token and positional embeddings
        tok = model.token_embed(x)
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)

        # Combine
        h = tok + pos

        # The model only has one layer, but we can extract from the first layer
        encoder_layer = model.transformer.layers[0]

        # Get self-attention module
        mha = encoder_layer.self_attn

        # We need to manually compute Q, K projections
        # For batch_first=True, h is (batch, seq, d_model)

        d_model = model.d_model
        n_heads = model.n_heads
        head_dim = d_model // n_heads

        # For PyTorch MHA, in_proj_weight is shape (3 * d_model, d_model)
        # and contains [W_q, W_k, W_v] concatenated
        in_proj_weight = mha.in_proj_weight
        in_proj_bias = mha.in_proj_bias

        q_weight = in_proj_weight[:d_model, :]
        k_weight = in_proj_weight[d_model:2*d_model, :]

        q_bias = in_proj_bias[:d_model] if in_proj_bias is not None else None
        k_bias = in_proj_bias[d_model:2*d_model] if in_proj_bias is not None else None

        # Project Q and K
        # (batch, seq, d_model) @ (d_model, d_model) -> (batch, seq, d_model)
        q = torch.nn.functional.linear(h, q_weight, q_bias)
        k = torch.nn.functional.linear(h, k_weight, k_bias)

        # Reshape for multi-head attention
        # (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
        q = q.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

        # Compute scaled dot-product attention scores
        # q: (batch, n_heads, seq, head_dim)
        # k: (batch, n_heads, head_dim, seq)
        # scores: (batch, n_heads, seq, seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        # Apply softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        return attn_weights

if __name__ == "__main__":
    pass

def plot_attention_heatmaps(model: ModularArithmeticTransformer,
                            dataset: torch.Tensor,
                            step: int,
                            condition: str,
                            save_dir: str = "results/figures/attention") -> None:
    """
    Generate attention heatmaps for the model's layers at a given training step.
    Averages attention across the dataset and plots a heatmap for each attention head.

    Args:
        model: Model at a specific checkpoint.
        dataset: Input dataset of shape (N, 2).
        step: Training step (used for title and filename).
        condition: String describing experimental condition (e.g., 'pure', 'severe_collapse').
        save_dir: Directory to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get attention weights averaged over the dataset
    # Shape: (batch, n_heads, seq_len, seq_len)
    attn_weights = extract_attention_weights(model, dataset)

    # Average across the batch
    # Shape: (n_heads, 2, 2)
    avg_attn = attn_weights.mean(dim=0).cpu().numpy()

    n_heads = model.n_heads

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for i in range(n_heads):
        ax = axes[i]
        sns.heatmap(avg_attn[i], annot=True, cmap="Blues", fmt=".3f",
                    xticklabels=['pos 0 (a)', 'pos 1 (b)'],
                    yticklabels=['pos 0 (a)', 'pos 1 (b)'],
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(f"Head {i}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    plt.suptitle(f"Attention Patterns ({condition} - Step {step})")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"attention_heatmap_{condition}_step{step}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def analyze_head_specialization(model: ModularArithmeticTransformer,
                                dataset: torch.Tensor) -> Dict[str, float]:
    """
    Analyze attention head specialization.
    For this 2-token task, we measure the specialization of each head towards
    attending to position 0 vs position 1, or self vs other.

    Args:
        model: Model at a specific checkpoint.
        dataset: Input dataset of shape (N, 2).

    Returns:
        Dict mapping head index to its specialization score.
    """
    attn_weights = extract_attention_weights(model, dataset)
    # Average across batch
    avg_attn = attn_weights.mean(dim=0).cpu().numpy()  # (n_heads, 2, 2)

    specialization_scores = {}
    for i in range(model.n_heads):
        # Measure tendency of pos 1 to attend to pos 0 vs pos 1
        # This captures the causal link information flow
        pos1_to_pos0 = avg_attn[i, 1, 0]
        pos1_to_pos1 = avg_attn[i, 1, 1]

        # Simple ratio or difference, we use the probability mass on pos 0 from pos 1
        specialization_scores[f"head_{i}_pos1_to_pos0"] = float(pos1_to_pos0)

    return specialization_scores


def compute_attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of the attention distribution.

    Args:
        attn_weights: Tensor of shape (batch, n_heads, seq_len, seq_len)

    Returns:
        Entropy tensor of shape (batch, n_heads, seq_len)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    # Entropy: -sum(p * log(p))
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
    return entropy

def track_attention_evolution(model_path_template: str,
                              steps: List[int],
                              dataset: torch.Tensor,
                              model_config: dict) -> Dict[str, List[float]]:
    """
    Track attention metrics (like entropy and specialization) over training steps.

    Args:
        model_path_template: String template for checkpoint path, e.g. 'results/pure/checkpoint_{step}.pt'
        steps: List of training steps to evaluate.
        dataset: Evaluation dataset.
        model_config: Dictionary containing model initialization kwargs.

    Returns:
        Dictionary mapping metric names to lists of values corresponding to the steps.
    """
    device = dataset.device
    model = ModularArithmeticTransformer(**model_config).to(device)

    history = {f"head_{i}_pos1_to_pos0": [] for i in range(model_config.get('n_heads', 4))}
    history["mean_entropy"] = []

    for step in steps:
        ckpt_path = model_path_template.format(step=step)
        if not os.path.exists(ckpt_path):
            continue

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        # Head specialization
        specs = analyze_head_specialization(model, dataset)
        for k, v in specs.items():
            history[k].append(v)

        # Entropy
        attn_weights = extract_attention_weights(model, dataset)
        entropy = compute_attention_entropy(attn_weights)
        # Average across batch, heads, and sequence length for a global metric
        history["mean_entropy"].append(entropy.mean().item())

    return history

def measure_attention_similarity(model1: ModularArithmeticTransformer,
                                 model2: ModularArithmeticTransformer,
                                 dataset: torch.Tensor) -> float:
    """
    Measure similarity (e.g. MSE) between attention patterns of two models.
    Useful for comparing models trained with different collapse levels.

    Args:
        model1: First model.
        model2: Second model.
        dataset: Input dataset.

    Returns:
        MSE between the averaged attention weights.
    """
    attn1 = extract_attention_weights(model1, dataset).mean(dim=0)
    attn2 = extract_attention_weights(model2, dataset).mean(dim=0)

    mse = torch.nn.functional.mse_loss(attn1, attn2).item()
    return float(mse)

def identify_grokking_circuits(model: ModularArithmeticTransformer, dataset: torch.Tensor, threshold: float = 0.8) -> List[int]:
    """
    Identify which heads form the 'grokking circuit'.
    For this modular addition task, grokking often coincides with specific heads
    strongly routing information from pos 1 to pos 0.

    Args:
        model: Fully trained model (post-grokking).
        dataset: Input dataset.
        threshold: The attention weight threshold to consider a head as part of the circuit.

    Returns:
        List of head indices that form the primary circuit.
    """
    specs = analyze_head_specialization(model, dataset)
    circuit_heads = []

    for i in range(model.n_heads):
        score = specs.get(f"head_{i}_pos1_to_pos0", 0.0)
        if score > threshold:
            circuit_heads.append(i)

    return circuit_heads
