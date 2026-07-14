import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Tuple
import os
import copy


def interpolate_1d(model: nn.Module,
                  loss_fn: Callable,
                  data: Tuple[torch.Tensor, torch.Tensor],
                  state_a: Dict[str, torch.Tensor],
                  state_b: Dict[str, torch.Tensor],
                  steps: int = 20,
                  alpha_range: Tuple[float, float] = (-0.5, 1.5)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolates weights between state_a and state_b and computes loss at each step.

    Args:
        model: PyTorch model.
        loss_fn: Loss function.
        data: Tuple of (inputs, targets).
        state_a: Starting state dict.
        state_b: Ending state dict.
        steps: Number of interpolation steps.
        alpha_range: Range of interpolation multiplier.

    Returns:
        alphas (np.ndarray), losses (np.ndarray)
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
    losses = []

    inputs, targets = data
    device = next(model.parameters()).device
    inputs, targets = inputs.to(device), targets.to(device)

    for alpha in alphas:
        # Interpolate state
        interp_state = {}
        for k in state_a.keys():
            if 'weight' in k or 'bias' in k:
                interp_state[k] = (1 - alpha) * state_a[k] + alpha * state_b[k]
            else:
                interp_state[k] = state_a[k]

        model.load_state_dict(interp_state)
        model.eval()

        with torch.no_grad():
            loss = loss_fn(model, inputs, targets)
            losses.append(loss.item())

    return alphas, np.array(losses)


def generate_random_direction(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Generates a random direction with the same shape as state_dict."""
    direction = {}
    for k, v in state_dict.items():
        if v.dtype.is_floating_point:
            direction[k] = torch.randn_like(v)
    return direction


def filter_normalize(direction: Dict[str, torch.Tensor],
                     state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Applies filter normalization (Li et al., 2018) to a direction vector.
    For each filter/row, normalizes the direction to have the same norm as the original weights.
    """
    norm_direction = {}
    for k in direction.keys():
        d = direction[k]
        w = state_dict[k]

        if d.dim() <= 1:
            # Bias or 1D tensor: just scale by norm
            d_norm = torch.norm(d)
            w_norm = torch.norm(w)
            if d_norm > 0:
                norm_direction[k] = d * (w_norm / d_norm)
            else:
                norm_direction[k] = d
        else:
            # 2D+ tensor (filters)
            # Flatten all dimensions except the first (output channels/filters)
            d_flat = d.view(d.size(0), -1)
            w_flat = w.view(w.size(0), -1)

            d_norms = torch.norm(d_flat, dim=1, keepdim=True)
            w_norms = torch.norm(w_flat, dim=1, keepdim=True)

            # Avoid division by zero
            d_norms = torch.where(d_norms > 1e-10, d_norms, torch.ones_like(d_norms))

            scale = (w_norms / d_norms).view([-1] + [1] * (d.dim() - 1))
            norm_direction[k] = d * scale

    return norm_direction


def plot_2d_landscape(model: nn.Module,
                      loss_fn: Callable,
                      data: Tuple[torch.Tensor, torch.Tensor],
                      center_state: Dict[str, torch.Tensor],
                      grid_size: int = 21,
                      limit: float = 1.0,
                      save_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plots a 2D loss landscape slice around center_state using filter-normalized random directions.

    Args:
        model: PyTorch model.
        loss_fn: Loss function.
        data: Tuple of (inputs, targets).
        center_state: State dict to center the plot on.
        grid_size: Resolution of the grid.
        limit: Max coordinate value for x and y axes.
        save_path: Path to save figure.

    Returns:
        X, Y, Z coordinates for the landscape.
    """
    # 1. Generate two random normalized directions
    dir_x = generate_random_direction(center_state)
    dir_y = generate_random_direction(center_state)

    dir_x = filter_normalize(dir_x, center_state)
    dir_y = filter_normalize(dir_y, center_state)

    # 2. Setup grid
    x = np.linspace(-limit, limit, grid_size)
    y = np.linspace(-limit, limit, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    inputs, targets = data
    device = next(model.parameters()).device
    inputs, targets = inputs.to(device), targets.to(device)

    # 3. Evaluate loss over grid
    for i in range(grid_size):
        for j in range(grid_size):
            alpha = X[i, j]
            beta = Y[i, j]

            # W' = W + alpha * dx + beta * dy
            eval_state = {}
            for k in center_state.keys():
                if k in dir_x:
                    eval_state[k] = center_state[k] + alpha * dir_x[k] + beta * dir_y[k]
                else:
                    eval_state[k] = center_state[k]

            model.load_state_dict(eval_state)
            model.eval()

            with torch.no_grad():
                loss = loss_fn(model, inputs, targets)
                Z[i, j] = loss.item()

    # 4. Plot
    plt.figure(figsize=(8, 6))

    # Log scale for contours often looks better
    contour = plt.contourf(X, Y, np.log(Z + 1e-8), levels=20, cmap='viridis')
    plt.colorbar(contour, label='Log Loss')
    plt.contour(X, Y, np.log(Z + 1e-8), levels=20, colors='k', alpha=0.3)

    plt.plot(0, 0, 'r*', markersize=15, label='Center')

    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('2D Loss Landscape')
    plt.legend()

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return X, Y, Z


def get_trajectory_coordinates(checkpoints: List[Dict],
                              center_state: Dict[str, torch.Tensor],
                              dir_x: Dict[str, torch.Tensor],
                              dir_y: Dict[str, torch.Tensor]) -> Tuple[List[float], List[float]]:
    """
    Projects a sequence of checkpoints onto the 2D plane defined by dir_x and dir_y.
    Uses least squares projection.
    """
    coords_x = []
    coords_y = []

    # Flatten base state and directions
    flat_base = torch.cat([v.flatten() for v in center_state.values() if v.dtype.is_floating_point])
    flat_dx = torch.cat([v.flatten() for v in dir_x.values()])
    flat_dy = torch.cat([v.flatten() for v in dir_y.values()])

    # Create projection matrix (2 x N)
    basis = torch.stack([flat_dx, flat_dy], dim=1) # (N, 2)
    # pseudo-inverse for projection: (B^T B)^-1 B^T
    pinv_basis = torch.linalg.pinv(basis) # (2, N)

    for ckpt in checkpoints:
        state = ckpt.get('model_state', ckpt)
        flat_state = torch.cat([state[k].flatten() for k in center_state.keys() if state[k].dtype.is_floating_point])

        diff = flat_state - flat_base
        coords = pinv_basis @ diff

        coords_x.append(coords[0].item())
        coords_y.append(coords[1].item())

    return coords_x, coords_y


def overlay_trajectory(X: np.ndarray,
                       Y: np.ndarray,
                       Z: np.ndarray,
                       coords_x: List[float],
                       coords_y: List[float],
                       save_path: str = None):
    """
    Overlays a projected trajectory onto a 2D landscape contour plot.
    """
    plt.figure(figsize=(8, 6))

    contour = plt.contourf(X, Y, np.log(Z + 1e-8), levels=20, cmap='viridis')
    plt.colorbar(contour, label='Log Loss')

    # Plot trajectory
    plt.plot(coords_x, coords_y, 'w.-', alpha=0.8, linewidth=2, markersize=8, label='Optimization Path')
    plt.plot(coords_x[0], coords_y[0], 'go', markersize=10, label='Start')
    plt.plot(coords_x[-1], coords_y[-1], 'ro', markersize=10, label='End')

    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('Optimization Trajectory on Loss Landscape')
    plt.legend()

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_landscapes(model: nn.Module,
                       loss_fn: Callable,
                       data: Tuple[torch.Tensor, torch.Tensor],
                       grokked_state: Dict[str, torch.Tensor],
                       collapsed_state: Dict[str, torch.Tensor],
                       save_path: str = None):
    """
    Generates side-by-side 1D landscape interpolations (from init to final)
    to compare grokked and collapsed geometries.
    Here we just interpolate from Grokked to Collapsed as a proxy.
    """
    alphas, losses = interpolate_1d(model, loss_fn, data, grokked_state, collapsed_state, steps=30, alpha_range=(-0.5, 1.5))

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, losses, 'b-', linewidth=2)
    plt.axvline(x=0, color='g', linestyle='--', label='Grokked State')
    plt.axvline(x=1, color='r', linestyle='--', label='Collapsed State')

    plt.xlabel('Interpolation Coefficient (alpha)')
    plt.ylabel('Loss')
    plt.title('1D Interpolation: Grokked to Collapsed')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
