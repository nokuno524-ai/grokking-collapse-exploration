import os
import json
import torch
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
import torch.nn.functional as F

try:
    from src.model import ModularArithmeticTransformer
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import ModularArithmeticTransformer

def extract_attention_patterns(model, x):
    """
    Extract attention weights from ModularArithmeticTransformer since need_weights=False
    in standard nn.TransformerEncoderLayer.
    """
    batch_size, seq_len = x.shape
    d_model = model.d_model
    n_heads = model.n_heads
    d_k = d_model // n_heads

    # Forward pass to get token + pos embeddings
    tok = model.token_embed(x)
    positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    pos = model.pos_embed(positions)
    h = tok + pos

    # For a 1-layer transformer
    layer = model.transformer.layers[0]

    # Pre-layer norm if there is one (PyTorch default often puts norm after or before)
    # Let's check PyTorch's default: norm1 is applied inside
    h_norm = layer.norm1(h)

    # The self-attention module
    attn = layer.self_attn

    # Q, K, V projections
    in_proj_weight = attn.in_proj_weight
    in_proj_bias = attn.in_proj_bias

    # Project
    qkv = F.linear(h_norm, in_proj_weight, in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    # Reshape for multi-head attention
    q = q.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)

    # Scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = torch.softmax(scores, dim=-1)

    return attn_weights

def compute_attention_entropy(attn_weights):
    """
    Compute Shannon entropy of attention distributions.
    attn_weights: (batch, n_heads, seq_len, seq_len)
    """
    # Entropy over the key dimension (last dim)
    # add small epsilon to avoid log(0)
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
    # Mean over batch and queries
    return entropy.mean(dim=(0, 2))

def identify_phase_transition(results_json_path):
    """
    Identify grokking step via discrete derivative of test accuracy.
    Returns -1 if max accuracy remains below 0.9.
    """
    with open(results_json_path, 'r') as f:
        data = json.load(f)

    history = data.get('history', [])
    if not history:
        return -1

    test_accs = [h['test_acc'] for h in history]
    steps = [h['step'] for h in history]

    max_acc = max(test_accs)
    if max_acc < 0.9:
        return -1

    acc_arr = np.array(test_accs)
    diff = np.diff(acc_arr)
    # Find max increase
    transition_idx = np.argmax(diff) + 1
    return steps[transition_idx]

def load_checkpoints_and_analyze(run_dir):
    """
    Load all checkpoints for a given run and track attention entropy over time.
    """
    ckpts = sorted(glob(os.path.join(run_dir, 'checkpoint_*.pt')),
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))

    entropies = []
    steps = []

    # Standard evaluation input (e.g. all pairs)
    prime = 59
    a = torch.arange(prime)
    b = torch.arange(prime)
    A, B = torch.meshgrid(a, b, indexing='ij')
    x = torch.stack([A.flatten(), B.flatten()], dim=1)

    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        step = ckpt.get('step', int(ckpt_path.split('_')[-1].split('.')[0]))

        # We need the config to init the model
        config = ckpt.get('config', {'prime': 59, 'd_model': 128, 'n_heads': 4, 'd_ff': 512})
        model = ModularArithmeticTransformer(
            prime=config.get('prime', 59),
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            d_ff=config.get('d_ff', 512)
        )
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        with torch.no_grad():
            attn = extract_attention_patterns(model, x)
            entropy = compute_attention_entropy(attn)
            entropies.append(entropy.numpy())
            steps.append(step)

    return np.array(steps), np.array(entropies)

def generate_multi_panel_figure(pure_dir, collapse_dir, output_prefix):
    """
    Create a comprehensive multi-panel figure.
    top row = attention heatmaps at early/mid/late steps for pure model
    bottom row = same for collapsed model
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    prime = 59
    # Sample input
    x = torch.tensor([[10, 20], [5, 15], [30, 40], [50, 2]])

    def plot_attention_row(run_dir, row_idx, title_prefix):
        ckpts = sorted(glob(os.path.join(run_dir, 'checkpoint_*.pt')),
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if not ckpts:
            return

        early_ckpt = ckpts[len(ckpts)//10] if len(ckpts) > 10 else ckpts[0]
        mid_ckpt = ckpts[len(ckpts)//2]
        late_ckpt = ckpts[-1]

        selected = [('Early', early_ckpt), ('Mid', mid_ckpt), ('Late', late_ckpt)]

        for col_idx, (stage, ckpt_path) in enumerate(selected):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            config = ckpt.get('config', {'prime': 59, 'd_model': 128, 'n_heads': 4, 'd_ff': 512})
            model = ModularArithmeticTransformer(
                prime=config.get('prime', 59),
                d_model=config.get('d_model', 128),
                n_heads=config.get('n_heads', 4),
                d_ff=config.get('d_ff', 512)
            )
            model.load_state_dict(ckpt['model_state'])
            model.eval()

            with torch.no_grad():
                attn = extract_attention_patterns(model, x)
                # Mean over batch, first head
                avg_attn = attn.mean(dim=0)[0].numpy()

            ax = axes[row_idx, col_idx]
            sns.heatmap(avg_attn, ax=ax, cmap="YlGnBu", vmin=0, vmax=1)
            ax.set_title(f"{title_prefix} - {stage}")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")

    plot_attention_row(pure_dir, 0, "Pure")
    plot_attention_row(collapse_dir, 1, "Collapsed")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300)
    plt.savefig(f"{output_prefix}.pdf")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pure_dir', type=str, default="results/pure/seed_42")
    parser.add_argument('--collapse_dir', type=str, default="results/severe_collapse/seed_42")
    parser.add_argument('--output', type=str, default="results/attention_evolution")
    args = parser.parse_args()

    if os.path.exists(args.pure_dir) and os.path.exists(args.collapse_dir):
        generate_multi_panel_figure(args.pure_dir, args.collapse_dir, args.output)
        print(f"Generated figures at {args.output}.png/pdf")
    else:
        print("Provided directories do not exist. Skipping figure generation.")
