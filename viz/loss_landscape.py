"""
Loss landscape visualization for grokking-collapse experiments.
Computes the test loss on a 2D grid defined by two random orthogonal directions
in the parameter space, producing contour plots of the loss landscape.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import copy

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ModularArithmeticTransformer
from src.data import generate_modular_arithmetic, DatasetConfig

def get_random_directions(model: torch.nn.Module) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate two random orthogonal directions (D1, D2) in the parameter space,
    normalized per filter/parameter.
    """
    d1 = []
    d2 = []

    # Extract weights and generate directions
    for p in model.parameters():
        # Direction 1
        d1_tensor = torch.randn_like(p)
        # Direction 2
        d2_tensor = torch.randn_like(p)

        # Normalize by norm of weights for scale invariance
        p_norm = p.norm() + 1e-10
        d1_tensor = d1_tensor * (p_norm / (d1_tensor.norm() + 1e-10))
        d2_tensor = d2_tensor * (p_norm / (d2_tensor.norm() + 1e-10))

        d1.append(d1_tensor)
        d2.append(d2_tensor)

    # Orthogonalize d2 w.r.t d1 (Gram-Schmidt)
    dot_product = sum((x * y).sum() for x, y in zip(d1, d2))
    d1_norm_sq = sum((x * x).sum() for x in d1)

    proj = dot_product / d1_norm_sq

    d2_ortho = []
    for x, y in zip(d1, d2):
        d2_ortho.append(y - proj * x)

    # Re-normalize d2_ortho
    d2_norm = sum((x * x).sum() for x in d2_ortho) ** 0.5
    d1_norm = d1_norm_sq ** 0.5

    for i in range(len(d1)):
        d1[i] = d1[i] / d1_norm
        d2_ortho[i] = d2_ortho[i] / d2_norm

    return d1, d2_ortho

def compute_loss_grid(model: torch.nn.Module,
                      inputs: torch.Tensor,
                      targets: torch.Tensor,
                      d1: List[torch.Tensor],
                      d2: List[torch.Tensor],
                      grid_size: int = 21,
                      alpha_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute loss on a 2D grid.
    W = W0 + a*D1 + b*D2
    """
    alphas = np.linspace(-alpha_max, alpha_max, grid_size)
    betas = np.linspace(-alpha_max, alpha_max, grid_size)

    X, Y = np.meshgrid(alphas, betas)
    Z = np.zeros_like(X)

    # Save original weights
    orig_weights = [p.clone().detach() for p in model.parameters()]
    device = next(model.parameters()).device
    inputs, targets = inputs.to(device), targets.to(device)

    model.eval()
    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                # Apply perturbation
                for p, w0, dx, dy in zip(model.parameters(), orig_weights, d1, d2):
                    p.data = w0 + a * dx.to(device) + b * dy.to(device)

                # Compute loss
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                Z[j, i] = loss.item()  # Note index order for contourf

    # Restore original weights
    for p, w0 in zip(model.parameters(), orig_weights):
        p.data = w0

    return X, Y, Z

def plot_loss_landscape(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                        ax: plt.Axes, title: str = "Loss Landscape"):
    """Plot contour of the loss landscape."""
    # Use log scale for contours if loss range is large
    if Z.max() / (Z.min() + 1e-10) > 100:
        levels = np.logspace(np.log10(Z.min() + 1e-5), np.log10(Z.max()), 20)
    else:
        levels = np.linspace(Z.min(), Z.max(), 20)

    contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
    ax.contour(X, Y, Z, levels=levels, colors='k', alpha=0.3, linewidths=0.5)

    # Mark the center (unperturbed model)
    ax.plot(0, 0, 'r*', markersize=10)

    ax.set_title(title)
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    return contour

def compare_landscapes(results_dir: Path, output_dir: Path, prime: int = 59):
    """
    Compare loss landscapes of pure vs collapsed models across training steps.
    """
    pure_dir = results_dir / "pure"
    severe_dir = results_dir / "severe_collapse"

    if not (pure_dir.exists() and severe_dir.exists()):
        print("Pure or severe_collapse directories not found.")
        return

    # Get test data (using standard pure generation config)
    data_config = DatasetConfig(prime=prime, seed=42)
    _, _, test_in, test_tgt = generate_modular_arithmetic(data_config)

    # Pick checkpoints
    steps_to_check = [10000, 30000, 50000]

    fig, axes = plt.subplots(2, len(steps_to_check), figsize=(5 * len(steps_to_check), 10))

    for col, step in enumerate(steps_to_check):
        # Pure model
        pure_ckpt_path = pure_dir / f"checkpoint_{step}.pt"
        if pure_ckpt_path.exists():
            ckpt = torch.load(pure_ckpt_path, map_location="cpu")
            model = ModularArithmeticTransformer(prime=prime)
            model.load_state_dict(ckpt["model_state"])

            d1, d2 = get_random_directions(model)
            X, Y, Z = compute_loss_grid(model, test_in, test_tgt, d1, d2, grid_size=11, alpha_max=20.0)

            plot_loss_landscape(X, Y, Z, axes[0, col], title=f"Pure (Step {step})")

        # Collapsed model
        severe_ckpt_path = severe_dir / f"checkpoint_{step}.pt"
        if severe_ckpt_path.exists():
            ckpt = torch.load(severe_ckpt_path, map_location="cpu")
            model = ModularArithmeticTransformer(prime=prime)
            model.load_state_dict(ckpt["model_state"])

            d1, d2 = get_random_directions(model)
            X, Y, Z = compute_loss_grid(model, test_in, test_tgt, d1, d2, grid_size=11, alpha_max=20.0)

            plot_loss_landscape(X, Y, Z, axes[1, col], title=f"Collapsed (Step {step})")

    plt.tight_layout()
    plt.savefig(output_dir / "loss_landscape_evolution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="viz_output")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing loss landscapes (this may take a minute)...")
    compare_landscapes(results_dir, output_dir)
    print("Done!")
