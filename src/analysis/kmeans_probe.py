"""
Mechanistic Probe: k-means signature in attention.
Tests if attention heads implement a Lloyd's k-means like step by measuring
the alignment between attention patterns and token representation cluster centroids.
"""

import torch
import math
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig, generate_modular_arithmetic


def get_dataloader():
    config = DatasetConfig(prime=59, train_fraction=0.3)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)
    dataset = torch.utils.data.TensorDataset(test_in, test_tgt)
    return torch.utils.data.DataLoader(dataset, batch_size=512)


def signature_from_tensors(h: torch.Tensor, attn_weights: torch.Tensor) -> List[float]:
    """
    Given representations (batch, seq, d_model) and attention weights
    (batch, n_heads, seq_len, seq_len), computes the k-means correlation signature.
    """
    n_heads = attn_weights.shape[1]
    head_alignments = []
    for head_idx in range(n_heads):
        head_attn = attn_weights[:, head_idx, :, :]

        dist = torch.norm(h.unsqueeze(2) - h.unsqueeze(1), dim=-1)

        flat_dist = dist.flatten()
        flat_attn = head_attn.flatten()

        flat_dist_c = flat_dist - flat_dist.mean()
        flat_attn_c = flat_attn - flat_attn.mean()

        if flat_dist_c.std() < 1e-6 or flat_attn_c.std() < 1e-6:
            corr = 0.0
        else:
            corr = (flat_dist_c * flat_attn_c).mean() / (flat_dist_c.std() * flat_attn_c.std())

        head_alignments.append(-corr.item())
    return head_alignments


def compute_kmeans_signature(
    model: ModularArithmeticTransformer,
    val_loader: torch.utils.data.DataLoader,
    k_clusters: int = 5
) -> np.ndarray:
    """
    Measures the k-means signature of attention heads.
    A high score means the attention weights strongly align with the
    token representation cluster centroids (like a k-means assignment step).
    """
    model.eval()

    layer = model.transformer.layers[0]
    in_proj_weight = layer.self_attn.in_proj_weight.detach()
    in_proj_bias = layer.self_attn.in_proj_bias.detach()
    d_model = model.d_model
    n_heads = model.n_heads
    head_dim = d_model // n_heads

    all_alignments = []

    with torch.no_grad():
        for batch_x, _ in val_loader:
            device = next(model.parameters()).device
            batch_x = batch_x.to(device)
            batch_size = batch_x.shape[0]

            tok = model.token_embed(batch_x)
            positions = torch.arange(2, device=device).unsqueeze(0).expand(batch_size, -1)
            pos = model.pos_embed(positions)
            h = tok + pos
            seq_len = h.size(1)

            qkv = torch.nn.functional.linear(h, in_proj_weight, in_proj_bias)
            qkv = qkv.reshape(batch_size, seq_len, 3, n_heads, head_dim)
            q, k_val, v = qkv.unbind(2)

            q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
            k_val = k_val.transpose(1, 2)

            scores = torch.matmul(q, k_val.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = torch.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)

            head_alignments = signature_from_tensors(h, attn_weights)
            all_alignments.append(head_alignments)

    # Average across batches
    mean_alignments = np.mean(all_alignments, axis=0)
    return mean_alignments


def analyze_checkpoints(results_dir: Path, output_dir: Path):
    """Analyze k-means signature evolution across all checkpoints in each severity condition."""
    severities = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    val_loader = get_dataloader()

    checkpoint_steps = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]

    results = {}

    for severity in severities:
        sev_dir = results_dir / severity
        if not sev_dir.exists():
            continue

        results[severity] = {'steps': [], 'kmeans_signatures': []}

        for step in checkpoint_steps:
            chk_path = sev_dir / f"checkpoint_{step}.pt"
            if not chk_path.exists():
                continue

            model = ModularArithmeticTransformer()
            state_dict = torch.load(chk_path, map_location="cpu", weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'model_state' in state_dict:
                state_dict = state_dict['model_state']

            # Clean keys if needed
            clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict)

            signature = compute_kmeans_signature(model, val_loader)

            results[severity]['steps'].append(step)
            results[severity]['kmeans_signatures'].append(signature)

    plot_kmeans_evolution(results, output_dir)
    return results

def plot_kmeans_evolution(results: Dict, output_dir: Path):
    """Plot k-means signature over time for each severity."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for severity, data in results.items():
        if not data['steps']:
            continue

        steps = data['steps']
        signatures = np.array(data['kmeans_signatures'])  # shape (num_steps, num_heads)

        plt.figure(figsize=(8, 5))

        num_heads = signatures.shape[1]
        for h in range(num_heads):
            plt.plot(steps, signatures[:, h], label=f'Head {h}', marker='o')

        plt.title(f"k-means Signature Evolution ({severity})")
        plt.xlabel("Training Step")
        plt.ylabel("k-means Signature Score")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"kmeans_signature_{severity}.png", dpi=150)
        plt.close()

    print(f"Saved k-means signature plots to {output_dir}")

if __name__ == "__main__":
    import sys
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else base_dir / "figures"
    analyze_checkpoints(base_dir, output_dir)
