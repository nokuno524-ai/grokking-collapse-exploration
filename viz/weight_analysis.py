import torch
import torch.nn as nn
import numpy as np
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import copy

# Add the project root to the sys path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ModularArithmeticTransformer

def compute_hessian_max_eigenvalue(model: nn.Module, loss_fn: Any, data: Tuple[torch.Tensor, torch.Tensor],
                                  max_iter: int = 50, tol: float = 1e-4) -> float:
    """
    Compute the dominant Hessian eigenvalue using power iteration.
    This gives a measure of sharpness in the loss landscape.
    """
    # Disable SDP flash attention which doesn't support double backward on CPU
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        model.eval()
        x, y = data

        # 1. Compute first gradients
        model.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)

        # Get parameters that require gradients
        params = [p for p in model.parameters() if p.requires_grad]

        # Compute gradients
        grads = torch.autograd.grad(loss, params, create_graph=True)

        # 2. Initialize random vector v
        v = [torch.randn_like(p) for p in params]
        # Normalize v
        norm = torch.sqrt(sum(torch.sum(vi ** 2) for vi in v))
        v = [vi / norm for vi in v]

        # 3. Power iteration
        eigenvalue = 0.0
        for i in range(max_iter):
            # Compute Hessian-vector product (Hv)
            # We compute d(grads * v) / d(params)

            # Dot product of grads and v
            grad_v_dot = sum(torch.sum(g * vi) for g, vi in zip(grads, v))

            # Gradient of the dot product gives Hv
            Hv = torch.autograd.grad(grad_v_dot, params, retain_graph=True)

            # Rayleigh quotient gives eigenvalue estimate: v^T H v / v^T v
            # Since v is normalized, this is just v^T (Hv)
            new_eigenvalue = sum(torch.sum(vi * hvi) for vi, hvi in zip(v, Hv)).item()

            # Update v
            norm = torch.sqrt(sum(torch.sum(hvi ** 2) for hvi in Hv))
            if norm.item() == 0:
                break

            v = [hvi / norm for hvi in Hv]

            # Check convergence
            if abs(new_eigenvalue - eigenvalue) < tol:
                eigenvalue = new_eigenvalue
                break

            eigenvalue = new_eigenvalue

        return eigenvalue

def get_filter_normalized_directions(model: nn.Module) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate two random directions for loss landscape contouring,
    scaled by the filter (layer) norms.
    """
    dir1 = []
    dir2 = []

    for p in model.parameters():
        if p.requires_grad:
            # Random direction
            d1 = torch.randn_like(p)
            d2 = torch.randn_like(p)

            # Normalize direction by filter norm
            if p.dim() >= 2:
                # For matrices, normalize per filter (output dimension)
                p_norm = p.norm(dim=tuple(range(1, p.dim())), keepdim=True)
                d1_norm = d1.norm(dim=tuple(range(1, d1.dim())), keepdim=True)
                d2_norm = d2.norm(dim=tuple(range(1, d2.dim())), keepdim=True)

                # Avoid division by zero
                d1_norm = torch.clamp(d1_norm, min=1e-10)
                d2_norm = torch.clamp(d2_norm, min=1e-10)

                d1 = d1 * (p_norm / d1_norm)
                d2 = d2 * (p_norm / d2_norm)
            else:
                # For vectors (biases), normalize whole vector
                p_norm = p.norm()
                d1_norm = d1.norm()
                d2_norm = d2.norm()

                if d1_norm > 1e-10: d1 = d1 * (p_norm / d1_norm)
                if d2_norm > 1e-10: d2 = d2 * (p_norm / d2_norm)

            dir1.append(d1)
            dir2.append(d2)

    return dir1, dir2

def plot_loss_landscape_contour(model: nn.Module, loss_fn: Any, data: Tuple[torch.Tensor, torch.Tensor],
                               grid_size: int = 11, scale: float = 1.0, save_path: Optional[str] = None) -> np.ndarray:
    """
    Evaluate the loss on a 2D grid around the current model parameters.
    Returns the loss grid.
    """
    model.eval()
    x, y = data

    dir1, dir2 = get_filter_normalized_directions(model)

    # Create grid
    alphas = np.linspace(-scale, scale, grid_size)
    betas = np.linspace(-scale, scale, grid_size)

    loss_grid = np.zeros((grid_size, grid_size))

    # Save original weights
    orig_weights = [p.data.clone() for p in model.parameters() if p.requires_grad]

    params = [p for p in model.parameters() if p.requires_grad]

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Update weights: w' = w + alpha * d1 + beta * d2
                for p, p0, d1, d2 in zip(params, orig_weights, dir1, dir2):
                    p.data = p0 + alpha * d1 + beta * d2

                # Evaluate loss
                logits = model(x)
                loss = loss_fn(logits, y).item()
                loss_grid[i, j] = loss

    # Restore original weights
    for p, p0 in zip(params, orig_weights):
        p.data = p0.clone()

    if save_path:
        plt.figure(figsize=(8, 6))
        X, Y = np.meshgrid(alphas, betas)
        # Log scale often looks better for loss landscapes
        Z = np.log1p(loss_grid.T) # Transpose so alpha is x-axis, beta is y-axis

        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Log Loss')
        plt.plot(0, 0, 'rx', markersize=10, label='Current Weights')
        plt.title('Loss Landscape Contour (Filter Normalized)')
        plt.xlabel('Direction 1')
        plt.ylabel('Direction 2')
        plt.legend()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    return loss_grid

def extract_weight_norm_trajectory(checkpoint_paths: List[str], steps: List[int]) -> Dict[str, Any]:
    """
    Extract weight norms for specific layers across checkpoints.
    """
    n_steps = len(checkpoint_paths)
    if n_steps == 0:
        return {}

    # We'll track specific key components
    target_keys = ['token_embed.weight', 'pos_embed.weight', 'output_head.weight']

    metrics = {
        'steps': steps,
        'total_norm': np.zeros(n_steps)
    }
    for k in target_keys:
        metrics[k] = np.zeros(n_steps)

    for i, path in enumerate(checkpoint_paths):
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            state = checkpoint['model_state']

            # Total norm
            total_sq = sum((t.float().norm().item() ** 2) for t in state.values())
            metrics['total_norm'][i] = np.sqrt(total_sq)

            # Specific layer norms
            for k in target_keys:
                if k in state:
                    metrics[k][i] = state[k].float().norm().item()

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return metrics

if __name__ == "__main__":
    # Test script with dummy checkpoint
    checkpoint_path = "tests/data/dummy_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print("Testing weight space analysis...")

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        config = checkpoint.get('config', {})
        model = ModularArithmeticTransformer(
            prime=config.get('prime', 59),
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            d_ff=config.get('d_ff', 512),
            n_layers=config.get('n_layers', 1)
        )
        model.load_state_dict(checkpoint['model_state'])

        # Disable flash attention on CPU for testing Hessian computation
        # (It throws "derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented")
        torch.backends.native_sdp_enable_flash = False
        torch.backends.native_sdp_enable_math = True
        torch.backends.native_sdp_enable_mem_efficient = False

        # Dummy data
        x = torch.randint(0, config.get('prime', 59), (16, 2))
        y = (x[:, 0] + x[:, 1]) % config.get('prime', 59)
        loss_fn = nn.CrossEntropyLoss()

        # Test Hessian max eigenvalue
        eig = compute_hessian_max_eigenvalue(model, loss_fn, (x, y), max_iter=5)
        print(f"Hessian Max Eigenvalue (est): {eig:.4f}")

        # Test Loss Landscape Contour
        os.makedirs("tests/output", exist_ok=True)
        loss_grid = plot_loss_landscape_contour(model, loss_fn, (x, y), grid_size=3, save_path="tests/output/dummy_landscape.png")
        print(f"Loss grid shape: {loss_grid.shape}")
        print("Saved loss landscape to tests/output/dummy_landscape.png")

        # Test Norm Trajectory
        traj = extract_weight_norm_trajectory([checkpoint_path, checkpoint_path], [0, 1000])
        print("Trajectory keys:", list(traj.keys()))
        print("Total norms:", traj['total_norm'])
    else:
        print("Run tests/generate_checkpoint.py first to create a dummy checkpoint.")