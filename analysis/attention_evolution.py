import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import json
from pathlib import Path
import math

from src.model import ModularArithmeticTransformer

def get_attention_entropies(model, probe_batch):
    """
    Computes per-head attention entropy over a fixed probe batch.
    Returns: entropy per head (n_heads,)
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings
        tok = model.token_embed(probe_batch)
        positions = torch.arange(2, device=probe_batch.device).unsqueeze(0).expand(probe_batch.shape[0], -1)
        pos = model.pos_embed(positions)
        h = tok + pos

        # We need to manually do the attention to get the weights
        # Or patch the transformer
        # Since it's a simple 1-layer transformer, let's extract the weights

        # Get self-attention weights
        layer = model.transformer.layers[0]
        # h shape: (batch, seq_len=2, d_model)
        batch_size, seq_len, d_model = h.shape

        # Pre-attention norm
        # PyTorch TransformerEncoderLayer does attention -> norm -> dropout -> add -> etc
        # Actually standard PyTorch layer does:
        # h2 = self.norm1(h) if self.norm_first else h
        # h2, attn_weights = self.self_attn(h2, h2, h2, ...)

        # Let's check how the model is configured.
        # default batch_first=True, norm_first=False
        h2 = h

        # We can just call self_attn directly and ask for weights
        # average_attn_weights=False gives per-head weights if supported,
        # but PyTorch multihead attention requires need_weights=True.
        # However, getting per-head weights from nn.MultiheadAttention
        # requires a trick or extracting the query/key/value manually.

        in_proj_weight = layer.self_attn.in_proj_weight
        in_proj_bias = layer.self_attn.in_proj_bias

        qkv = F.linear(h2, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        head_dim = d_model // layer.self_attn.num_heads
        q = q.view(batch_size, seq_len, layer.self_attn.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, layer.self_attn.num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores, dim=-1) # (batch, n_heads, seq_len, seq_len)

        # Calculate entropy per head per batch element, then average over batch and seq_len
        # entropy of categorical distribution p: -sum(p * log(p + eps))
        entropy = -torch.sum(attn * torch.log(attn + 1e-9), dim=-1) # (batch, n_heads, seq_len)
        # mean over batch and seq_len (positions)
        mean_entropy = entropy.mean(dim=(0, 2)) # (n_heads,)

    return mean_entropy.cpu().numpy()

def analyze_attention_evolution(results_dir, output_dir, device="cpu"):
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Conditions
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    # Probe batch (same for all evaluations)
    torch.manual_seed(42)
    # 1024 random pairs
    probe_batch = torch.randint(0, 59, (1024, 2)).to(device)

    for condition in conditions:
        cond_dir = results_path / condition
        if not cond_dir.exists():
            continue

        print(f"Processing condition: {condition}")

        checkpoints = list(cond_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            print(f"  No checkpoints found for {condition}")
            continue

        # Sort checkpoints by step
        def get_step(p):
            try:
                return int(p.stem.split('_')[1])
            except:
                return -1

        checkpoints.sort(key=get_step)

        steps = []
        entropies = []

        for ckpt_path in checkpoints:
            step = get_step(ckpt_path)

            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                config = ckpt['config']

                model = ModularArithmeticTransformer(
                    prime=config.get('prime', 59),
                    d_model=config.get('d_model', 128),
                    n_heads=config.get('n_heads', 4),
                    d_ff=config.get('d_ff', 512),
                    n_layers=config.get('n_layers', 1)
                ).to(device)

                # Check for module prefix
                state_dict = ckpt['model_state']
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                model.load_state_dict(state_dict)

                head_entropy = get_attention_entropies(model, probe_batch)
                steps.append(step)
                entropies.append(head_entropy)
            except Exception as e:
                print(f"  Failed loading {ckpt_path}: {e}")

        if not steps:
            continue

        steps = np.array(steps)
        entropies = np.array(entropies) # (n_steps, n_heads)

        # Plot curves
        plt.figure(figsize=(10, 6))
        for h in range(entropies.shape[1]):
            plt.plot(steps, entropies[:, h], label=f'Head {h}', marker='o', markersize=4)

        plt.title(f'Attention Entropy Evolution - {condition}')
        plt.xlabel('Training Step')
        plt.ylabel('Shannon Entropy')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(out_path / f'entropy_curves_{condition}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot heatmap
        plt.figure(figsize=(12, 4))
        # Transpose so heads are y-axis and steps are x-axis
        im = plt.imshow(entropies.T, aspect='auto', cmap='viridis', origin='lower',
                        extent=[steps[0], steps[-1], -0.5, entropies.shape[1]-0.5])
        plt.colorbar(im, label='Entropy')
        plt.title(f'Attention Entropy Heatmap - {condition}')
        plt.xlabel('Training Step')
        plt.ylabel('Head Index')
        plt.yticks(range(entropies.shape[1]))
        plt.savefig(out_path / f'entropy_heatmap_{condition}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    analyze_attention_evolution("results", "analysis/attention")
    print("Attention evolution analysis complete.")
