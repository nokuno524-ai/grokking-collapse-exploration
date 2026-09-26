import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two sets of representations.
    Linear CKA version.

    Args:
        X (torch.Tensor): First representation tensor of shape (N, D1).
        Y (torch.Tensor): Second representation tensor of shape (N, D2).

    Returns:
        float: The CKA similarity score between 0 and 1.
    """
    # X: (N, D1), Y: (N, D2)
    N = X.shape[0]

    # Center X and Y
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    # Compute dot products
    XX = X_centered @ X_centered.T
    YY = Y_centered @ Y_centered.T

    # Frobenius norm
    num = torch.trace(XX @ YY)
    den = torch.sqrt(torch.trace(XX @ XX) * torch.trace(YY @ YY))

    if den == 0:
        return 0.0

    return (num / den).item()


def compute_effective_rank(X: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute the effective rank of a representation matrix X.
    Effective rank is the entropy of the normalized singular values.

    Args:
        X (torch.Tensor): Representation tensor of shape (N, D).
        eps (float): Small epsilon to prevent log(0).

    Returns:
        float: The effective rank.
    """
    # X: (N, D)
    s = torch.linalg.svdvals(X)
    s = s / (s.sum() + eps)
    entropy = -(s * torch.log(s + eps)).sum()
    return torch.exp(entropy).item()

def compute_probe_accuracy(X: torch.Tensor, y: torch.Tensor, lr: float = 1e-2, epochs: int = 100) -> float:
    """
    Fit a linear probe (logistic regression) on representations X to predict labels y.
    Returns the accuracy.

    Args:
        X (torch.Tensor): Representation tensor of shape (N, D).
        y (torch.Tensor): Target labels of shape (N,).
        lr (float): Learning rate for the probe.
        epochs (int): Number of epochs to train the probe.

    Returns:
        float: Accuracy of the probe on the provided data.
    """
    # X: (N, D)
    # y: (N,)

    device = X.device
    num_classes = len(torch.unique(y))
    D = X.shape[1]

    model = nn.Linear(D, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Simple training loop
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()

    return acc

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import torch
import torch.nn as nn

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from src.checkpoint import load_run
from src.model import ModularArithmeticTransformer
from src.data import generate_modular_arithmetic, DatasetConfig

@dataclass
class GeometryMetrics:
    step: int
    condition: str
    layer_ckas: Dict[str, float]
    layer_ranks: Dict[str, float]
    layer_probes: Dict[str, float]

def extract_representations(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Extract hidden representations from each layer of the model.

    Args:
        model (nn.Module): The transformer model.
        dataloader (DataLoader): DataLoader providing the dataset.
        device (torch.device): Compute device.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping layer names to their concatenated representation tensors.
    """
    model.eval()

    reps = {
        'tok_emb': [],
        'pos_emb': [],
        'transformer_out': [],
        'pool_out': [],
        'targets': []
    }

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.shape[0]

            # Follow model.py logic
            tok = model.token_embed(inputs) # (batch, 2, d_model)

            positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0).expand(batch_size, -1)
            pos = model.pos_embed(positions) # (batch, 2, d_model)

            h = tok + pos

            # Transformer
            h_trans = model.transformer(h) # (batch, 2, d_model)
            h_norm = model.ln(h_trans)

            h_pool = h_norm.mean(dim=1) # (batch, d_model)

            reps['tok_emb'].append(tok.reshape(batch_size, -1).cpu())
            reps['pos_emb'].append(pos.reshape(batch_size, -1).cpu())
            reps['transformer_out'].append(h_trans.reshape(batch_size, -1).cpu())
            reps['pool_out'].append(h_pool.cpu())
            reps['targets'].append(targets.cpu())

    # Concatenate
    return {k: torch.cat(v, dim=0) for k, v in reps.items()}

def analyze_geometry(results_dir: Path, output_dir: Path, seed: int = 42, prime: int = 59):
    """
    Compute CKA, Effective Rank, and Probe Accuracy over training for all conditions.

    Args:
        results_dir (Path): The directory containing condition-specific subdirectories with checkpoints.
        output_dir (Path): The directory to save generated metrics and plots.
        seed (int): The random seed for data generation.
        prime (int): The prime modulus parameter.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Identify conditions
    conditions = ["pure", "low_collapse", "medium_collapse", "severe_collapse", "high_collapse"]

    # Generate common test data
    data_config = DatasetConfig(prime=prime, seed=seed)
    _, _, test_in, test_tgt = generate_modular_arithmetic(data_config)
    test_dataset = torch.utils.data.TensorDataset(test_in, test_tgt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 1. Load the final grokked reference model (pure run)
    pure_dir = results_dir / "pure"
    if not pure_dir.exists():
        raise FileNotFoundError(f"Pure run directory not found: {pure_dir}")

    try:
        ref_sd, ref_cfg = load_run(pure_dir)
    except FileNotFoundError:
        print(f"No checkpoint found in {pure_dir}, skipping geometry analysis")
        return

    ref_model = ModularArithmeticTransformer(
        prime=ref_cfg.get("prime", prime),
        d_model=ref_cfg.get("d_model", 128),
        n_heads=ref_cfg.get("n_heads", 4),
        d_ff=ref_cfg.get("d_ff", 512),
        n_layers=ref_cfg.get("n_layers", 1),
    ).to(device)
    ref_model.load_state_dict(ref_sd)
    ref_reps = extract_representations(ref_model, test_loader, device)

    # 2. Analyze all checkpoints across all conditions
    all_metrics = []

    for condition in conditions:
        cond_dir = results_dir / condition
        if not cond_dir.exists():
            continue

        print(f"Analyzing condition: {condition}")

        # Get all checkpoints
        import re
        ckpts = sorted(cond_dir.glob("checkpoint_*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[-1]))

        for ckpt_path in ckpts:
            step = int(re.findall(r"\d+", ckpt_path.name)[-1])

            try:
                sd, cfg = load_run(cond_dir, step=step)
            except Exception as e:
                print(f"Error loading {ckpt_path}: {e}")
                continue

            model = ModularArithmeticTransformer(
                prime=cfg.get("prime", prime),
                d_model=cfg.get("d_model", 128),
                n_heads=cfg.get("n_heads", 4),
                d_ff=cfg.get("d_ff", 512),
                n_layers=cfg.get("n_layers", 1),
            ).to(device)
            model.load_state_dict(sd)

            reps = extract_representations(model, test_loader, device)
            targets = reps['targets']

            # Compute metrics
            layers = ['tok_emb', 'transformer_out', 'pool_out']
            layer_ckas = {}
            layer_ranks = {}
            layer_probes = {}

            for layer in layers:
                X = reps[layer]
                Y = ref_reps[layer]

                # CKA
                layer_ckas[layer] = compute_cka(X, Y)

                # Effective Rank
                layer_ranks[layer] = compute_effective_rank(X)

                # Probe Accuracy (subset for speed if needed, but test set is small)
                layer_probes[layer] = compute_probe_accuracy(X, targets)

            metrics = GeometryMetrics(
                step=step,
                condition=condition,
                layer_ckas=layer_ckas,
                layer_ranks=layer_ranks,
                layer_probes=layer_probes
            )
            all_metrics.append(metrics)

    # 3. Save and Plot
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    if HAS_PANDAS:
        records = []
        for m in all_metrics:
            for layer in ['tok_emb', 'transformer_out', 'pool_out']:
                records.append({
                    'step': m.step,
                    'condition': m.condition,
                    'layer': layer,
                    'cka': m.layer_ckas[layer],
                    'rank': m.layer_ranks[layer],
                    'probe_acc': m.layer_probes[layer]
                })
        df = pd.DataFrame(records)
        df.to_csv(output_dir / "geometry_metrics.csv", index=False)

        # Generate Onset Map (Divergence Localization)
        # Find earliest step where CKA drops below threshold (e.g., 0.8) for non-pure conditions
        threshold = 0.8
        onset_records = []

        for condition in conditions:
            if condition == "pure": continue
            cond_df = df[df['condition'] == condition]
            if cond_df.empty: continue

            for layer in ['tok_emb', 'transformer_out', 'pool_out']:
                layer_df = cond_df[cond_df['layer'] == layer]
                # Find first step where CKA < threshold
                drop_df = layer_df[layer_df['cka'] < threshold]
                if not drop_df.empty:
                    onset_step = drop_df['step'].min()
                else:
                    onset_step = np.nan

                onset_records.append({
                    'condition': condition,
                    'layer': layer,
                    'onset_step': onset_step
                })

        onset_df = pd.DataFrame(onset_records)
        onset_df.to_csv(output_dir / "collapse_onset_map.csv", index=False)

        # Plotting
        if HAS_PLT:
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))

            layers = ['tok_emb', 'transformer_out', 'pool_out']
            metrics = ['cka', 'rank', 'probe_acc']
            metric_titles = ['CKA to Pure Ref', 'Effective Rank', 'Probe Accuracy']

            for i, layer in enumerate(layers):
                for j, metric in enumerate(metrics):
                    ax = axes[i, j]
                    layer_df = df[df['layer'] == layer]
                    sns.lineplot(data=layer_df, x='step', y=metric, hue='condition', ax=ax)
                    ax.set_title(f"{layer}: {metric_titles[j]}")
                    if metric == 'cka':
                        ax.axhline(threshold, color='r', linestyle='--', alpha=0.5)
                        ax.set_ylim(0, 1.05)
                    elif metric == 'probe_acc':
                        ax.set_ylim(0, 1.05)

            plt.tight_layout()
            plt.savefig(output_dir / "geometry_trajectories.png")
            plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/geometry"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prime", type=int, default=59)
    args = parser.parse_args()

    analyze_geometry(args.results_dir, args.output_dir, args.seed, args.prime)
