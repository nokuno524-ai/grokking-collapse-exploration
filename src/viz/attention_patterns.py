import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.model import ModularArithmeticTransformer

CONDITIONS = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

def get_attention_weights(model, x):
    """
    Compute attention weights manually for ModularArithmeticTransformer
    since nn.TransformerEncoderLayer uses need_weights=False by default.
    """
    batch_size = x.shape[0]

    with torch.no_grad():
        tok = model.token_embed(x)
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = model.pos_embed(positions)
        emb = tok + pos

        layer = model.transformer.layers[0]
        attn = layer.self_attn

        d_model = model.d_model

        q_weight = attn.in_proj_weight[:d_model, :]
        k_weight = attn.in_proj_weight[d_model:2*d_model, :]

        q_bias = attn.in_proj_bias[:d_model] if attn.in_proj_bias is not None else 0
        k_bias = attn.in_proj_bias[d_model:2*d_model] if attn.in_proj_bias is not None else 0

        Q = F.linear(emb, q_weight, q_bias)
        K = F.linear(emb, k_weight, k_bias)

        n_heads = model.n_heads
        head_dim = d_model // n_heads

        Q = Q.view(batch_size, 2, n_heads, head_dim).transpose(1, 2)
        K = K.view(batch_size, 2, n_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1) # (batch, n_heads, seq_len, seq_len)

    return weights

def load_checkpoint(ckpt_path):
    model = ModularArithmeticTransformer()
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'model_state' in state_dict:
            state_dict = state_dict['model_state']

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        return model
    except Exception as e:
        print(f"Failed to load {ckpt_path}: {e}")
        return None

def plot_attention_patterns(results_dir="results"):
    results_path = Path(results_dir)
    out_dir = Path("src/viz/attention")
    out_dir.mkdir(parents=True, exist_ok=True)

    x_test = torch.randint(0, 59, (50, 2))
    target_steps = [10000, 30000, 50000] # Early, Mid, Late

    # We want a plot for each step, comparing conditions
    for step in target_steps:
        fig, axes = plt.subplots(len(CONDITIONS), 4, figsize=(16, 4 * len(CONDITIONS)))
        fig.suptitle(f"Attention Patterns at Step {step}", fontsize=16)

        for i, condition in enumerate(CONDITIONS):
            ckpt_path = results_path / condition / f"checkpoint_{step}.pt"
            if not ckpt_path.exists():
                print(f"Checkpoint not found for {condition} at step {step}")
                continue

            model = load_checkpoint(ckpt_path)
            if model is None:
                continue

            weights = get_attention_weights(model, x_test)
            # Average across batch
            avg_weights = weights.mean(dim=0).numpy() # (n_heads, 2, 2)

            for h in range(model.n_heads): # n_heads
                ax = axes[i, h]
                sns.heatmap(avg_weights[h], annot=True, fmt=".2f", cmap="Blues", ax=ax,
                           xticklabels=["pos0", "pos1"], yticklabels=["pos0", "pos1"],
                           vmin=0, vmax=1)
                if i == 0:
                    ax.set_title(f"Head {h+1}")
                if h == 0:
                    ax.set_ylabel(condition, fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        out_path = out_dir / f"attention_patterns_step_{step}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    plot_attention_patterns()
