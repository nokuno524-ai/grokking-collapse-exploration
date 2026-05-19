import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
from typing import Tuple, Optional, Callable, Dict, Any

def get_random_direction(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Generate a random direction vector matching the model's parameters.
    The direction is normalized filter-wise (layer-wise) to match the scale
    of the original weights, following Li et al. (2018).
    """
    direction = {}
    for name, param in model.named_parameters():
        # Generate random values from standard normal distribution
        rand_d = torch.randn_like(param)

        # Filter-wise normalization (Frobenius norm)
        # Scale the random direction to have the same norm as the parameter
        d_norm = rand_d.norm()
        p_norm = param.norm()

        if d_norm > 0:
            rand_d = rand_d * (p_norm / d_norm)

        direction[name] = rand_d

    return direction

def evaluate_model_at_point(
    model: nn.Module,
    base_weights: Dict[str, torch.Tensor],
    directions: Tuple[Dict[str, torch.Tensor], ...],
    alphas: Tuple[float, ...],
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """
    Evaluate the model loss at a specific point in the parameter space:
    base_weights + alpha1 * dir1 + alpha2 * dir2 + ...
    """
    # Temporarily modify model weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in base_weights:
                new_weight = base_weights[name].clone()
                for d, a in zip(directions, alphas):
                    if name in d:
                        new_weight += a * d[name]
                param.copy_(new_weight)

    # Evaluate loss
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            total_loss += loss.item()
            total_samples += inputs.size(0)

    # Restore original weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in base_weights:
                param.copy_(base_weights[name])

    return total_loss / total_samples if total_samples > 0 else float('inf')

def compute_1d_loss_slice(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    alpha_range: Tuple[float, float, int] = (-1.0, 1.0, 21)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, torch.Tensor]]:
    """
    Compute loss along a 1D random direction slice.
    alpha_range: (min, max, num_points)
    Returns: alphas, losses, direction
    """
    alphas = np.linspace(*alpha_range)
    losses = np.zeros_like(alphas)

    # Save base weights
    base_weights = {name: param.clone().detach() for name, param in model.named_parameters()}

    # Generate random direction
    direction = get_random_direction(model)

    for i, alpha in enumerate(alphas):
        losses[i] = evaluate_model_at_point(model, base_weights, (direction,), (alpha,), dataloader, device)

    return alphas, losses, direction

def compute_2d_loss_landscape(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    alpha_range: Tuple[float, float, int] = (-1.0, 1.0, 21),
    beta_range: Tuple[float, float, int] = (-1.0, 1.0, 21)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Compute loss along a 2D plane defined by two random directions.
    Returns: Alphas (2D), Betas (2D), Losses (2D), direction1, direction2
    """
    alpha_vals = np.linspace(*alpha_range)
    beta_vals = np.linspace(*beta_range)

    A, B = np.meshgrid(alpha_vals, beta_vals)
    Losses = np.zeros_like(A)

    # Save base weights
    base_weights = {name: param.clone().detach() for name, param in model.named_parameters()}

    # Generate random directions
    dir1 = get_random_direction(model)
    dir2 = get_random_direction(model)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            alpha = A[i, j]
            beta = B[i, j]
            Losses[i, j] = evaluate_model_at_point(
                model, base_weights, (dir1, dir2), (alpha, beta), dataloader, device
            )

    return A, B, Losses, dir1, dir2

def plot_1d_slice(alphas: np.ndarray, losses: np.ndarray, output_path: str, title: str = "1D Loss Slice"):
    """Plot 1D loss slice."""
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, losses, 'b-', linewidth=2)
    plt.xlabel('Step Size (α)')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_2d_contour(
    A: np.ndarray,
    B: np.ndarray,
    Losses: np.ndarray,
    output_path: str,
    title: str = "2D Loss Landscape",
    levels: int = 30,
    log_scale: bool = True
):
    """Plot publication-quality 2D contour of loss landscape."""
    plt.figure(figsize=(10, 8))

    if log_scale:
        # Avoid log(0)
        Z = np.log10(Losses - Losses.min() + 1e-8)
    else:
        Z = Losses

    contour = plt.contourf(A, B, Z, levels=levels, cmap='viridis')
    plt.colorbar(contour, label='Log10(Loss)' if log_scale else 'Loss')

    # Plot center point (original weights)
    plt.plot(0, 0, 'r*', markersize=15, label='Original Model')

    plt.xlabel('Direction 1 (α)')
    plt.ylabel('Direction 2 (β)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate loss landscapes from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
    parser.add_argument("--data-config", type=str, required=True, help="Path to a dummy config to generate data or pure condition")
    parser.add_argument("--output-dir", type=str, default="analysis/loss_landscape", help="Where to save plots")
    parser.add_argument("--type", type=str, choices=["1d", "2d", "both"], default="both")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config_dict = ckpt.get("config", {})

    try:
        from src.model import ModularArithmeticTransformer
        from src.data import generate_modular_arithmetic, DatasetConfig
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("Could not import model and data. Run from repo root.")
        exit(1)

    model = ModularArithmeticTransformer(
        prime=config_dict.get("prime", 59),
        d_model=config_dict.get("d_model", 128),
        n_heads=config_dict.get("n_heads", 4),
        d_ff=config_dict.get("d_ff", 512),
        n_layers=config_dict.get("n_layers", 1),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    print("Model loaded successfully.")

    data_config = DatasetConfig(
        prime=config_dict.get("prime", 59),
        train_fraction=config_dict.get("train_fraction", 0.3),
        collapse_level=config_dict.get("collapse_level", 0.0),
        seed=config_dict.get("seed", 42),
    )
    _, _, test_in, test_tgt = generate_modular_arithmetic(data_config)
    test_loader = DataLoader(TensorDataset(test_in, test_tgt), batch_size=512)

    if args.type in ["1d", "both"]:
        print("Computing 1D slice...")
        alphas, losses, _ = compute_1d_loss_slice(model, test_loader, device)
        plot_1d_slice(alphas, losses, os.path.join(args.output_dir, "loss_1d.png"))

    if args.type in ["2d", "both"]:
        print("Computing 2D landscape (this may take a while)...")
        A, B, Losses, _, _ = compute_2d_loss_landscape(model, test_loader, device)
        plot_2d_contour(A, B, Losses, os.path.join(args.output_dir, "loss_2d.png"))

    print(f"Saved plots to {args.output_dir}")
