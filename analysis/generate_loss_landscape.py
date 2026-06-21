import torch
import torch.nn.functional as F
import copy
import numpy as np
import matplotlib.pyplot as plt
import os

def create_random_direction(model, device):
    """
    Creates a random direction vector normalized filter-wise
    to match the parameter scales of the model (Li et al., 2018).
    """
    direction = []
    for p in model.parameters():
        # Generate random values from standard normal
        d = torch.randn_like(p, device=device)

        # Filter-wise normalization
        if p.dim() <= 1:
            # Bias or LayerNorm: just normalize the whole vector
            d.mul_(p.norm() / (d.norm() + 1e-10))
        else:
            # Weights: normalize along the output dimension (dim 0)
            for i in range(p.shape[0]):
                d[i].mul_(p[i].norm() / (d[i].norm() + 1e-10))

        direction.append(d)
    return direction

def evaluate_loss(model, dataloader, device):
    """Evaluates the average cross-entropy loss over a dataloader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction='sum')
            total_loss += loss.item()
            total_samples += x.shape[0]

    return total_loss / total_samples if total_samples > 0 else float('inf')

def compute_loss_landscape(model, dataloader, device, steps=10, scale=1.0):
    """
    Computes a 2D grid of losses by perturbing the model parameters
    along two random directions.
    """
    dir1 = create_random_direction(model, device)
    dir2 = create_random_direction(model, device)

    # Save original weights
    orig_weights = [p.clone() for p in model.parameters()]

    alphas = np.linspace(-scale, scale, steps)
    betas = np.linspace(-scale, scale, steps)

    loss_grid = np.zeros((steps, steps))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            # Apply perturbation
            for p, p_orig, d1, d2 in zip(model.parameters(), orig_weights, dir1, dir2):
                p.data = p_orig.data + a * d1 + b * d2

            loss_grid[i, j] = evaluate_loss(model, dataloader, device)

    # Restore original weights
    for p, p_orig in zip(model.parameters(), orig_weights):
        p.data = p_orig.data

    return alphas, betas, loss_grid

def plot_landscape(alphas, betas, loss_grid, save_path):
    """Plots the 2D loss landscape as a contour plot."""
    A, B = np.meshgrid(alphas, betas)

    plt.figure(figsize=(8, 6))

    # We often plot log(loss) for better visibility if differences are extreme
    log_loss = np.log1p(loss_grid)

    cp = plt.contourf(A, B, log_loss.T, levels=30, cmap='viridis')
    plt.colorbar(cp, label='Log(1 + Loss)')

    plt.contour(A, B, log_loss.T, levels=30, colors='black', alpha=0.3, linewidths=0.5)

    # Mark the center (unperturbed model)
    plt.plot(0, 0, 'r*', markersize=15, label="Original Model")

    plt.title("2D Loss Landscape Contour")
    plt.xlabel("Direction 1 (α)")
    plt.ylabel("Direction 2 (β)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    from src.model import ModularArithmeticTransformer
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cpu")
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64, n_layers=1).to(device)

    # Dummy data
    x = torch.randint(0, 11, (100, 2))
    y = (x[:, 0] + x[:, 1]) % 11
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)

    print("Computing loss landscape... (this may take a few seconds)")
    alphas, betas, loss_grid = compute_loss_landscape(model, dataloader, device, steps=5, scale=0.5)

    plot_landscape(alphas, betas, loss_grid, "visualizations/dummy_loss_landscape.png")
    print("Saved loss landscape to visualizations/dummy_loss_landscape.png")
