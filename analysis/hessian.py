import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict
import os


def power_iteration(model: nn.Module,
                   loss_fn: Callable,
                   data: Tuple[torch.Tensor, torch.Tensor],
                   num_iterations: int = 20,
                   tol: float = 1e-4) -> Tuple[float, torch.Tensor]:
    """
    Estimates the dominant (largest) eigenvalue of the Hessian using power iteration.
    Optimized for memory efficiency by computing Hessian-vector products without forming the Hessian.

    Args:
        model: PyTorch model.
        loss_fn: Function that computes loss given model and data.
        data: Tuple of (inputs, targets).
        num_iterations: Maximum number of power iterations.
        tol: Tolerance for early stopping.

    Returns:
        Tuple of (max_eigenvalue, principal_eigenvector_flat).
    """
    device = next(model.parameters()).device
    inputs, targets = data
    inputs, targets = inputs.to(device), targets.to(device)

    # 1. Forward pass to get loss
    model.eval()
    loss = loss_fn(model, inputs, targets)

    # 2. Get first derivatives (gradients)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Initialize random vector v
    v = [torch.randn_like(p).to(device) for p in params]

    # Normalize v
    v_norm = torch.sqrt(sum(torch.sum(vi**2) for vi in v))
    v = [vi / v_norm for vi in v]

    eigenvalue = 0.0

    for i in range(num_iterations):
        # 3. Compute Hessian-vector product (Hv)
        # Hv = d/dw (grads^T * v)
        grad_v_dot = sum(torch.sum(g * vi) for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(grad_v_dot, params, retain_graph=True)

        # 4. Update eigenvalue estimate (Rayleigh quotient): lambda = v^T * Hv
        prev_eigenvalue = eigenvalue
        eigenvalue = sum(torch.sum(vi * hvi).item() for vi, hvi in zip(v, Hv))

        # 5. Normalize Hv to get next v
        v_norm = torch.sqrt(sum(torch.sum(hvi**2) for hvi in Hv))
        if v_norm.item() < 1e-10:
            break

        v = [hvi / v_norm for hvi in Hv]

        # Check convergence
        if abs(eigenvalue - prev_eigenvalue) < tol:
            break

    # Flatten eigenvector for return
    eigenvector = torch.cat([vi.flatten() for vi in v])

    return eigenvalue, eigenvector


def compute_top_k_eigenvalues(model: nn.Module,
                             loss_fn: Callable,
                             data: Tuple[torch.Tensor, torch.Tensor],
                             k: int = 1,
                             num_iterations: int = 20) -> Tuple[List[float], List[torch.Tensor]]:
    """
    Estimates top-k Hessian eigenvalues using power iteration with deflation.
    Note: Full Lanczos is complex; deflation provides a reasonable approximation.

    Args:
        model: PyTorch model.
        loss_fn: Loss function.
        data: Tuple of (inputs, targets).
        k: Number of top eigenvalues to compute.
        num_iterations: Number of power iterations per eigenvalue.

    Returns:
        List of eigenvalues, List of flattened eigenvectors.
    """
    device = next(model.parameters()).device
    inputs, targets = data
    inputs, targets = inputs.to(device), targets.to(device)

    model.eval()
    loss = loss_fn(model, inputs, targets)

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)

    eigenvalues = []
    eigenvectors = []

    # Store deflated directions
    found_vectors = []

    for _ in range(k):
        # Initialize random vector v
        v = [torch.randn_like(p).to(device) for p in params]

        # Orthogonalize against previously found eigenvectors
        for u in found_vectors:
            dot_uv = sum(torch.sum(vi * ui) for vi, ui in zip(v, u))
            v = [vi - dot_uv * ui for vi, ui in zip(v, u)]

        v_norm = torch.sqrt(sum(torch.sum(vi**2) for vi in v))
        v = [vi / v_norm for vi in v]

        eigenvalue = 0.0

        for _ in range(num_iterations):
            # Hessian-vector product
            grad_v_dot = sum(torch.sum(g * vi) for g, vi in zip(grads, v))
            Hv = list(torch.autograd.grad(grad_v_dot, params, retain_graph=True))

            # Deflation: subtract previously found eigenvalues/vectors
            for lambda_i, u in zip(eigenvalues, found_vectors):
                dot_uv = sum(torch.sum(vi * ui) for vi, ui in zip(v, u))
                Hv = [hvi - lambda_i * dot_uv * ui for hvi, ui in zip(Hv, u)]

            eigenvalue = sum(torch.sum(vi * hvi).item() for vi, hvi in zip(v, Hv))

            v_norm = torch.sqrt(sum(torch.sum(hvi**2) for hvi in Hv))
            if v_norm.item() < 1e-10:
                break

            v = [hvi / v_norm for hvi in Hv]

            # Orthogonalize again for stability
            for u in found_vectors:
                dot_uv = sum(torch.sum(vi * ui) for vi, ui in zip(v, u))
                v = [vi - dot_uv * ui for vi, ui in zip(v, u)]
            v_norm = torch.sqrt(sum(torch.sum(vi**2) for vi in v))
            v = [vi / v_norm for vi in v]

        eigenvalues.append(eigenvalue)
        found_vectors.append(v)
        eigenvectors.append(torch.cat([vi.flatten() for vi in v]))

    return eigenvalues, eigenvectors


def track_sharpness(checkpoints: List[Dict],
                   model: nn.Module,
                   loss_fn: Callable,
                   data: Tuple[torch.Tensor, torch.Tensor]) -> List[float]:
    """
    Tracks the maximum Hessian eigenvalue (sharpness) across checkpoints.

    Args:
        checkpoints: List of checkpoint states.
        model: Model instance to load weights into.
        loss_fn: Loss function.
        data: Evaluation data.

    Returns:
        List of maximum eigenvalues.
    """
    sharpness_trajectory = []

    for ckpt in checkpoints:
        state = ckpt.get('model_state', ckpt)
        model.load_state_dict(state)

        # Power iteration for top eigenvalue
        eig_val, _ = power_iteration(model, loss_fn, data)
        sharpness_trajectory.append(eig_val)

    return sharpness_trajectory


def estimate_hessian_rank(eigenvalues: List[float], threshold: float = 1e-3) -> int:
    """
    Estimates the effective rank of the Hessian based on a list of eigenvalues.

    Args:
        eigenvalues: List of computed top eigenvalues.
        threshold: Minimum value to consider an eigenvalue non-zero.

    Returns:
        Estimated rank (count of eigenvalues above threshold).
    """
    return sum(1 for val in eigenvalues if abs(val) > threshold)


def plot_eigenvalue_spectrum(eigenvalues: List[float], save_path: str = None):
    """
    Plots the spectrum of the computed top eigenvalues.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue Magnitude')
    plt.title('Hessian Eigenvalue Spectrum')
    plt.grid(True, alpha=0.3)

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_landscape_geometry(grokked_sharpness: List[float],
                              collapsed_sharpness: List[float],
                              save_path: str = None):
    """
    Compares the sharpness (max eigenvalue) trajectories.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(grokked_sharpness, label='Grokked', color='blue', alpha=0.8)
    if collapsed_sharpness:
        plt.plot(collapsed_sharpness, label='Collapsed', color='red', alpha=0.8)

    plt.xlabel('Checkpoint Index')
    plt.ylabel('Max Eigenvalue (Sharpness)')
    plt.title('Landscape Geometry: Sharpness Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
