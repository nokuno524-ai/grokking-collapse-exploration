import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def extract_attention_weights(model, inputs):
    """
    Extract attention weights directly from the nn.TransformerEncoderLayer.
    Since we use batch_first=True, we have to handle the shape correctly.
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings
        tok = model.token_embed(inputs)
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)

        h = tok + pos

        # Get the first encoder layer
        encoder_layer = model.transformer.layers[0]

        # nn.MultiheadAttention returns (attn_output, attn_output_weights)
        # when need_weights=True.
        # However, calling encoder_layer directly doesn't give us weights.
        # We must call self_attn manually.

        # self_attn expects (seq_len, batch, embed_dim) if batch_first=False
        # Our model has batch_first=True, so it expects (batch, seq_len, embed_dim)
        query = h
        key = h
        value = h

        attn_output, attn_weights = encoder_layer.self_attn(
            query, key, value,
            need_weights=True,
            average_attn_weights=False  # Get per-head weights
        )

        # Shape is usually (batch, num_heads, target_seq_len, source_seq_len)
        return attn_weights

def plot_attention_heatmaps(attn_weights, save_path):
    """
    Plots attention heatmaps for each head.
    attn_weights shape expected: (num_heads, seq_len, seq_len)
    """
    # Just take the first item in the batch
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]

    num_heads = attn_weights.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))

    if num_heads == 1:
        axes = [axes]

    for i in range(num_heads):
        sns.heatmap(
            attn_weights[i].cpu().numpy(),
            ax=axes[i],
            cmap='viridis',
            vmin=0, vmax=1,
            annot=True, fmt=".2f"
        )
        axes[i].set_title(f"Head {i+1}")
        axes[i].set_xlabel("Key Position")
        axes[i].set_ylabel("Query Position")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    from src.model import ModularArithmeticTransformer

    # Create a dummy model
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4, d_ff=512, n_layers=1)

    # Dummy input (batch=1, seq_len=2)
    inputs = torch.tensor([[10, 20]])

    weights = extract_attention_weights(model, inputs)
    os.makedirs("visualizations", exist_ok=True)
    plot_attention_heatmaps(weights, "visualizations/dummy_attention_heatmap.png")
    print(f"Generated heatmap for dummy model at visualizations/dummy_attention_heatmap.png")
