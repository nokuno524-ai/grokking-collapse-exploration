import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def extract_attention_patterns(model, data):
    """
    Extract attention patterns from the model for the given data.
    Since ModularArithmeticTransformer uses nn.TransformerEncoderLayer, we manually
    compute Q, K to get attention weights.

    Args:
        model: ModularArithmeticTransformer instance
        data: Input tensor of shape (batch_size, seq_len)

    Returns:
        attention_weights: Tensor of shape (batch_size, n_heads, seq_len, seq_len)
    """
    batch_size = data.shape[0]
    seq_len = data.shape[1]
    d_model = model.d_model
    n_heads = model.n_heads
    head_dim = d_model // n_heads

    with torch.no_grad():
        # Get input embeddings + positional embeddings
        tok = model.token_embed(data)
        positions = torch.arange(seq_len, device=data.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)
        x = tok + pos

        # Get self-attention parameters
        attn_layer = model.transformer.layers[0].self_attn
        in_proj_weight = attn_layer.in_proj_weight
        in_proj_bias = attn_layer.in_proj_bias

        # Project x to Q, K, V
        # in_proj_weight is shape (3 * d_model, d_model)
        # in_proj_bias is shape (3 * d_model)
        qkv = F.linear(x, in_proj_weight, in_proj_bias)

        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

    return attn_weights


def compute_attention_entropy(attention_weights):
    """
    Compute entropy of attention distributions to measure focus vs diffusion.

    Args:
        attention_weights: Tensor of shape (batch_size, n_heads, seq_len, seq_len)

    Returns:
        entropy: Tensor of shape (batch_size, n_heads, seq_len)
    """
    # attention_weights is a probability distribution over the last dimension
    # Add epsilon to prevent log(0)
    eps = 1e-10
    entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=-1)
    return entropy


def track_attention_specialization(attention_weights_across_checkpoints):
    """
    Track how attention entropy changes across training checkpoints.

    Args:
        attention_weights_across_checkpoints: List of attention_weights tensors

    Returns:
        avg_entropy_per_head: List of tensors, each shape (n_heads,)
    """
    avg_entropies = []
    for attn_weights in attention_weights_across_checkpoints:
        entropy = compute_attention_entropy(attn_weights)
        # Average over batch and seq_len dimensions to get per-head entropy
        avg_entropy = entropy.mean(dim=(0, 2))
        avg_entropies.append(avg_entropy)

    return avg_entropies


def visualize_attention_evolution(avg_entropies, steps, output_path="attention_entropy_heatmap.pdf"):
    """
    Generate a heatmap of attention entropy over time.

    Args:
        avg_entropies: List of tensors shape (n_heads,), length num_checkpoints
        steps: List of integer training steps
        output_path: Path to save the plot
    """
    # Convert to 2D numpy array: shape (n_heads, num_checkpoints)
    data = torch.stack(avg_entropies, dim=1).numpy()

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        data,
        cmap="viridis",
        xticklabels=steps,
        yticklabels=[f"Head {i}" for i in range(data.shape[0])]
    )

    # Improve x-axis labels (show fewer if there are many)
    if len(steps) > 10:
        for ind, label in enumerate(ax.get_xticklabels()):
            if ind % (len(steps) // 10) == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

    plt.title("Attention Entropy Evolution (Lower = More Specialized)")
    plt.xlabel("Training Step")
    plt.ylabel("Attention Head")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
