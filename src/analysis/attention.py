"""
Utility to compute and plot attention pattern evolution metrics:
- Per-layer, per-head attention entropy
- Head specialization scores
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


def calculate_metrics_from_weights(attn_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes entropy and specialization directly from attention weights.
    attn_weights: (batch, n_heads, seq_len, seq_len)
    """
    eps = 1e-10
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean(dim=(0, 2))
    specialization = attn_weights.var(dim=-1).mean(dim=(0, 2))
    return entropy, specialization


def compute_attention_metrics(
    model: ModularArithmeticTransformer,
    val_loader: torch.utils.data.DataLoader
) -> Dict[str, np.ndarray]:
    """
    Extract attention metrics for the single transformer layer.
    Reconstructs Q, K, V manually since need_weights=False is hardcoded in
    PyTorch's TransformerEncoderLayer.
    """
    model.eval()
    all_entropies = []
    all_specializations = []

    layer = model.transformer.layers[0]
    in_proj_weight = layer.self_attn.in_proj_weight.detach()
    in_proj_bias = layer.self_attn.in_proj_bias.detach()
    out_proj_weight = layer.self_attn.out_proj.weight.detach()
    d_model = model.d_model
    n_heads = model.n_heads
    head_dim = d_model // n_heads

    with torch.no_grad():
        for batch_x, _ in val_loader:
            device = next(model.parameters()).device
            batch_x = batch_x.to(device)
            batch_size = batch_x.shape[0]

            # Forward pass up to attention
            tok = model.token_embed(batch_x)
            positions = torch.arange(2, device=device).unsqueeze(0).expand(batch_size, -1)
            pos = model.pos_embed(positions)
            h = tok + pos

            seq_len = h.size(1)

            # Linear projection
            qkv = torch.nn.functional.linear(h, in_proj_weight, in_proj_bias)

            # Split Q, K, V
            qkv = qkv.reshape(batch_size, seq_len, 3, n_heads, head_dim)
            q, k, v = qkv.unbind(2)

            # Compute attention scores
            q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
            k = k.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = torch.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)

            entropy, specialization = calculate_metrics_from_weights(attn_weights)
            all_entropies.append(entropy)
            all_specializations.append(specialization)

    # Aggregate over batches
    mean_entropy = torch.stack(all_entropies).mean(dim=0).cpu().numpy()
    mean_specialization = torch.stack(all_specializations).mean(dim=0).cpu().numpy()

    return {
        "entropy": mean_entropy,
        "specialization": mean_specialization
    }

def analyze_checkpoints(results_dir: Path, output_dir: Path):
    """Analyze attention evolution across all checkpoints in each severity condition."""
    severities = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

    val_loader = get_dataloader()

    checkpoint_steps = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]

    results = {}

    for severity in severities:
        sev_dir = results_dir / severity
        if not sev_dir.exists():
            continue

        results[severity] = {'steps': [], 'entropies': [], 'specializations': []}

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

            metrics = compute_attention_metrics(model, val_loader)

            results[severity]['steps'].append(step)
            results[severity]['entropies'].append(metrics['entropy'])
            results[severity]['specializations'].append(metrics['specialization'])

    plot_attention_evolution(results, output_dir)
    return results

def plot_attention_evolution(results: Dict, output_dir: Path):
    """Plot entropy and specialization over time for each severity."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for severity, data in results.items():
        if not data['steps']:
            continue

        steps = data['steps']
        entropies = np.array(data['entropies'])  # shape (num_steps, num_heads)
        specializations = np.array(data['specializations'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        num_heads = entropies.shape[1]
        for h in range(num_heads):
            ax1.plot(steps, entropies[:, h], label=f'Head {h}', marker='o')
            ax2.plot(steps, specializations[:, h], label=f'Head {h}', marker='o')

        ax1.set_title(f"Attention Entropy ({severity})")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Entropy")
        ax1.legend()

        ax2.set_title(f"Head Specialization ({severity})")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Variance")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_dir / f"attention_evolution_{severity}.png", dpi=150)
        plt.close()

    print(f"Saved attention evolution plots to {output_dir}")

if __name__ == "__main__":
    import sys
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else base_dir / "figures"
    analyze_checkpoints(base_dir, output_dir)
