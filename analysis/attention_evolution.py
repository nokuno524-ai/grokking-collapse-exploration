import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict, List, Tuple

def get_attention_entropy(model, prime: int, device: str = 'cpu') -> torch.Tensor:
    """
    Computes per-head attention entropy over all inputs.
    Returns:
        entropies: shape (n_heads, seq_len)
    """
    model.eval()
    model.to(device)

    # Generate all pairs
    all_pairs = [(a, b) for a in range(prime) for b in range(prime)]
    x = torch.tensor(all_pairs, device=device) # (p*p, 2)

    with torch.no_grad():
        tok = model.token_embed(x)
        positions = torch.arange(2, device=device).unsqueeze(0).expand(x.shape[0], -1)
        pos = model.pos_embed(positions)
        h = tok + pos

        layer = model.transformer.layers[0]
        # need_weights=True returns (batch, n_heads, seq_len, seq_len) if average_attn_weights=False
        _, attn_weights = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)

        # attn_weights is (p*p, n_heads, seq_len, seq_len)
        # Compute entropy over the key dimension (last dimension)
        # H = - sum(p * log(p))
        eps = 1e-10
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1) # (p*p, n_heads, seq_len)

        # Average over all input pairs
        avg_entropy = entropy.mean(dim=0) # (n_heads, seq_len)

    return avg_entropy

def classify_head_context_matching(model, prime: int, device: str = 'cpu') -> torch.Tensor:
    """
    Classifies context-matching behavior (n-gram interpolation framework) for seq_len=2.
    Measures how much each head attends to position 0 vs position 1, averaged over inputs.
    Returns:
        pos_attn_weights: shape (n_heads, seq_len, seq_len)
    """
    model.eval()
    model.to(device)

    all_pairs = [(a, b) for a in range(prime) for b in range(prime)]
    x = torch.tensor(all_pairs, device=device)

    with torch.no_grad():
        tok = model.token_embed(x)
        positions = torch.arange(2, device=device).unsqueeze(0).expand(x.shape[0], -1)
        pos = model.pos_embed(positions)
        h = tok + pos

        layer = model.transformer.layers[0]
        _, attn_weights = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)

        # attn_weights is (p*p, n_heads, seq_len, seq_len)
        # Average over all input pairs
        avg_weights = attn_weights.mean(dim=0) # (n_heads, seq_len, seq_len)

    return avg_weights

def analyze_attention_evolution(history_dirs: List[str], model_cls, prime: int, output_dir: str):
    """
    Plots head specialization timelines and identifies transitions.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Process history
    # Assume history_dirs is a list of run directories (e.g., pure, noise, collapse)

    for run_dir in history_dirs:
        run_name = os.path.basename(run_dir)
        ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*.pt")), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

        if not ckpts:
            continue

        model = model_cls(prime=prime)

        steps = []
        entropies = []
        context_match = []

        for ckpt_path in ckpts:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            step = ckpt['step']

            ent = get_attention_entropy(model, prime, device).cpu()
            match = classify_head_context_matching(model, prime, device).cpu()

            steps.append(step)
            entropies.append(ent)
            context_match.append(match)

        entropies = torch.stack(entropies) # (steps, n_heads, seq_len)
        context_match = torch.stack(context_match) # (steps, n_heads, seq_len, seq_len)

        n_heads = entropies.shape[1]

        # Plot entropy over time
        plt.figure(figsize=(10, 6))
        for h in range(n_heads):
            # Average entropy over query positions
            plt.plot(steps, entropies[:, h, :].mean(dim=1).numpy(), label=f'Head {h}')
        plt.title(f'Attention Entropy over Time ({run_name})')
        plt.xlabel('Step')
        plt.ylabel('Average Entropy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'attention_entropy_{run_name}.png'))
        plt.close()

        # Plot context matching (attn to pos 0 from pos 1) over time
        plt.figure(figsize=(10, 6))
        for h in range(n_heads):
            # Attention from pos 1 to pos 0 (how much does output depend on input 'a')
            plt.plot(steps, context_match[:, h, 1, 0].numpy(), label=f'Head {h} (1->0)')
        plt.title(f'Context Matching (Attn to pos 0 from pos 1) over Time ({run_name})')
        plt.xlabel('Step')
        plt.ylabel('Attention Weight')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'context_matching_{run_name}.png'))
        plt.close()
