import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import json
import glob
import numpy as np

# Adjust sys path so we can import src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ModularArithmeticTransformer

def load_config(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
        return data.get("config", {})

def compute_attention_weights(model, x):
    # Pass inputs through embeddings
    token_embeds = model.token_embed(x)

    seq_len = x.size(1)
    pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
    pos_embeds = model.pos_embed(pos_ids)

    embeds = token_embeds + pos_embeds

    all_attn_weights = []

    for layer in model.transformer.layers:
        self_attn = layer.self_attn

        in_proj_weight = self_attn.in_proj_weight
        in_proj_bias = self_attn.in_proj_bias

        d_model = self_attn.embed_dim
        n_heads = self_attn.num_heads
        head_dim = d_model // n_heads

        # Project
        qkv = F.linear(embeds, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        q = q.view(q.size(0), q.size(1), n_heads, head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), n_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        # Apply softmax to get weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        all_attn_weights.append(attn_weights)

        # Assuming residual connection and feedforward here?
        # For extracting attention, we just need to pass the embeds to the next layer if there are multiple.
        # But we would need the true outputs of the attention + mlp.
        # Given this is just extracting attention weights, let's just do a proper forward pass
        embeds = layer(embeds)

    # return shape: (n_layers, batch, n_heads, seq_len, seq_len)
    return torch.stack(all_attn_weights)

def compute_attention_entropy(attn_weights):
    # attn_weights shape: (n_layers, batch, n_heads, seq_len, seq_len)
    # Entropy per query token: -sum(p * log(p + eps))
    eps = 1e-9
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
    # Average over batch and seq_len to get a scalar per head per layer
    return entropy.mean(dim=(1, 3))

def plot_attention_patterns(checkpoint_path, config_path, output_dir="viz/output"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load config and model
    config = load_config(config_path)

    prime = config.get("prime", 59)
    d_model = config.get("d_model", 128)
    n_heads = config.get("n_heads", 4)
    d_ff = config.get("d_ff", 512)
    n_layers = config.get("n_layers", 1)

    model = ModularArithmeticTransformer(
        prime=prime,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

    # Remove 'module.' prefix if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_state_dict[k.replace('module.', '')] = v

    model.load_state_dict(clean_state_dict)
    model.eval()

    # Generate a dummy input. From model.py, sequence length is 2 (positions 'a' and 'b').
    # Let's use arbitrary numbers from 0 to prime-1.
    x = torch.tensor([[10, 15]])

    with torch.no_grad():
        attn_weights = compute_attention_weights(model, x)

    # Plot heatmaps
    n_layers = attn_weights.size(0)
    n_heads = attn_weights.size(2)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4*n_heads, 4*n_layers))

    # Ensure axes is 2D array
    if n_layers == 1 and n_heads == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes[None, :]
    elif n_heads == 1:
        axes = axes[:, None]

    for l in range(n_layers):
        for h in range(n_heads):
            # We take the first example in the batch
            head_attn = attn_weights[l, 0, h].numpy()

            ax = axes[l, h]
            im = ax.imshow(head_attn, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"L{l+1} H{h+1}")
            if l == n_layers - 1:
                ax.set_xlabel("Key Position")
            if h == 0:
                ax.set_ylabel("Query Position")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path / "attention_patterns.png")
    plt.close()

    # Compute entropy
    entropy = compute_attention_entropy(attn_weights)

    # Plot entropy
    fig, axes = plt.subplots(n_layers, 1, figsize=(6, 4*n_layers))
    if n_layers == 1:
        axes = [axes]

    for l in range(n_layers):
        ax = axes[l]
        ax.bar(range(1, n_heads+1), entropy[l].numpy())
        ax.set_title(f"Layer {l+1} Attention Entropy per Head")
        ax.set_xlabel("Head")
        ax.set_ylabel("Entropy")
        ax.set_xticks(range(1, n_heads+1))

    plt.tight_layout()
    plt.savefig(out_path / "attention_entropy.png")
    plt.close()

if __name__ == "__main__":
    # Just an example path. Handle the case where they might not exist.
    cp_path = "results/pure/checkpoint_50000.pt"
    cfg_path = "results/pure/results.json"

    if os.path.exists(cp_path) and os.path.exists(cfg_path):
        plot_attention_patterns(cp_path, cfg_path)
    else:
        # Fallback to the first available checkpoint in pure
        checkpoints = sorted(glob.glob("results/pure/checkpoint_*.pt"))
        if checkpoints and os.path.exists(cfg_path):
            plot_attention_patterns(checkpoints[-1], cfg_path)
        else:
            print("Could not find required files for attention plotting.")
