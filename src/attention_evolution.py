import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
import os

def extract_attention_weights(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Extract attention weights from the ModularArithmeticTransformer for a given input batch.
    Returns tensor of shape (batch, n_heads, seq_len, seq_len).
    """
    model.eval()
    with torch.no_grad():
        tok = model.token_embed(inputs)
        batch_size = inputs.shape[0]

        # Dynamically infer sequence lengths via inputs.shape[1] per memory guidelines
        seq_len = inputs.shape[1]

        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)

        h = tok + pos

        # The transformer is nn.TransformerEncoder, containing layers
        layer = model.transformer.layers[0]

        # We need to extract the attention weights.
        # layer.self_attn is MultiheadAttention.
        # Signature: forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, ...)
        # Output: (attn_output, attn_output_weights)

        # nn.MultiheadAttention expects (seq_len, batch, d_model) if batch_first=False
        batch_first = getattr(layer.self_attn, 'batch_first', False)

        if not batch_first:
            h = h.transpose(0, 1) # (seq_len, batch, d_model)

        attn_out, attn_weights = layer.self_attn(
            h, h, h,
            need_weights=True,
            average_attn_weights=False
        )

        # attn_weights shape is (batch, n_heads, seq_len, seq_len) if average_attn_weights=False
        return attn_weights

def plot_attention_evolution(
    model_paths: List[str],
    steps: List[int],
    sample_inputs: torch.Tensor,
    model_class,
    model_kwargs: dict,
    save_path: str
):
    """
    Plot attention heatmaps for key checkpoints.

    Args:
        model_paths: List of paths to model checkpoints (.pt files)
        steps: List of corresponding step numbers
        sample_inputs: Tensor of inputs to evaluate attention on, shape (batch, seq_len)
        model_class: The model class (ModularArithmeticTransformer)
        model_kwargs: Initialization args for the model
        save_path: Where to save the plot
    """
    n_checkpoints = len(model_paths)
    n_heads = model_kwargs.get("n_heads", 4)

    fig, axes = plt.subplots(n_heads, n_checkpoints, figsize=(4*n_checkpoints, 4*n_heads))
    if n_heads == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_checkpoints == 1:
        axes = np.expand_dims(axes, axis=1)

    for j, (path, step) in enumerate(zip(model_paths, steps)):
        model = model_class(**model_kwargs)

        # Wrap torch.load with weights_only try/except per memory
        try:
            checkpoint = torch.load(path, weights_only=True)
        except Exception:
            checkpoint = torch.load(path, weights_only=False)

        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        # Average attention across the batch
        attn_weights = extract_attention_weights(model, sample_inputs)
        avg_attn = attn_weights.mean(dim=0).cpu().numpy() # (n_heads, seq_len, seq_len)

        for i in range(n_heads):
            ax = axes[i, j]
            sns.heatmap(avg_attn[i], ax=ax, cmap="Blues", vmin=0, vmax=1, cbar=False)
            if i == 0:
                ax.set_title(f"Step {step}")
            if j == 0:
                ax.set_ylabel(f"Head {i+1}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    print("Testing attention extraction...")
    from src.model import ModularArithmeticTransformer
    model = ModularArithmeticTransformer()
    x = torch.randint(0, 59, (4, 2))
    weights = extract_attention_weights(model, x)
    print("Extracted weights shape:", weights.shape)
