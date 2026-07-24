import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
import json
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ModularArithmeticTransformer
from src.data import generate_modular_arithmetic

def get_weights(model):
    """Extract all weights into a single flat vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_weights(model, weights):
    """Set weights from a single flat vector."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(weights[offset:offset + numel].view_as(p.data))
        offset += numel

def compute_loss(model, dataloader, device):
    """Compute test loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    return total_loss / total_samples

def get_random_directions(model, seed=42):
    """Generate two orthogonal random directions with same norm as weights."""
    torch.manual_seed(seed)

    # Extract weights
    weights = get_weights(model)
    norm = weights.norm()

    # Direction 1
    d1 = torch.randn_like(weights)
    d1 = d1 / d1.norm() * norm

    # Direction 2
    d2 = torch.randn_like(weights)
    d2 = d2 - torch.dot(d2, d1) / torch.dot(d1, d1) * d1 # Orthogonalize
    d2 = d2 / d2.norm() * norm

    return d1, d2

def generate_landscape_for_checkpoint(checkpoint_path, grid_size=11, scale=0.5, device="cpu"):
    """Generate 2D loss landscape for a single checkpoint."""
    # Load state
    state = torch.load(checkpoint_path, map_location=device)

    # Determine config if available
    config = state.get('config', {})
    prime = config.get('prime', 59)
    d_model = config.get('d_model', 128)
    n_heads = config.get('n_heads', 4)
    d_ff = config.get('d_ff', 512)

    # Load model
    model = ModularArithmeticTransformer(
        prime=prime,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=1
    ).to(device)

    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    elif 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    # Get test data
    from src.data import DatasetConfig
    dataset_cfg = DatasetConfig(prime=prime, train_fraction=0.3, seed=42)
    train_inputs, train_targets, test_inputs, test_targets = generate_modular_arithmetic(dataset_cfg)
    test_loader = DataLoader(TensorDataset(test_inputs, test_targets), batch_size=512)

    # Directions and base weights
    w_base = get_weights(model)
    d1, d2 = get_random_directions(model)

    # Grid
    alpha = np.linspace(-scale, scale, grid_size)
    beta = np.linspace(-scale, scale, grid_size)
    A, B = np.meshgrid(alpha, beta)

    losses = np.zeros((grid_size, grid_size))

    print(f"Computing loss grid for {checkpoint_path}...")
    for i in range(grid_size):
        for j in range(grid_size):
            a, b = A[i, j], B[i, j]

            # Perturb weights
            w_new = w_base + a * d1 + b * d2
            set_weights(model, w_new)

            # Compute loss
            losses[i, j] = compute_loss(model, test_loader, device)

    # Reset model to original weights
    set_weights(model, w_base)

    return A, B, losses

def main():
    parser = argparse.ArgumentParser(description="Generate loss landscape video.")
    parser.add_argument("--condition-dir", type=str, default="results/pure", help="Directory with checkpoints")
    parser.add_argument("--output", type=str, default="viz/loss_landscape.mp4", help="Output video path")
    parser.add_argument("--grid-size", type=int, default=11, help="Grid size for contour (NxN)")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale for random directions")

    args = parser.parse_args()

    cond_dir = Path(args.condition_dir)
    if not cond_dir.exists():
        print(f"Error: Directory {cond_dir} not found.")
        return

    # Get all checkpoints
    checkpoints = sorted([p for p in cond_dir.glob("checkpoint_*.pt")],
                        key=lambda p: int(p.stem.split('_')[1]))

    if not checkpoints:
        print(f"No checkpoints found in {cond_dir}")
        return

    # Read config to get grokking step for reference
    grokking_step = None
    try:
        with open(cond_dir / "results.json", "r") as f:
            res = json.load(f)
            grokking_step = res.get("grokking_step")
    except:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Store landscapes
    landscapes = []
    steps = []

    # Subsample checkpoints if there are too many to keep runtime reasonable
    if len(checkpoints) > 20:
        indices = np.linspace(0, len(checkpoints)-1, 20).astype(int)
        checkpoints = [checkpoints[i] for i in indices]

    # For testing, just take the first 3 if running interactively or in tests
    if os.environ.get("TEST_RUN", "0") == "1":
        checkpoints = checkpoints[:3]
        args.grid_size = 5

    for ckpt in checkpoints:
        step = int(ckpt.stem.split('_')[1])
        steps.append(step)
        A, B, L = generate_landscape_for_checkpoint(ckpt, grid_size=args.grid_size, scale=args.scale, device=device)
        landscapes.append((A, B, L))

    # Create animation
    fig, ax = plt.subplots(figsize=(8, 6))

    def animate(i):
        ax.clear()
        A, B, L = landscapes[i]
        step = steps[i]

        # Log scale contour
        contour = ax.contourf(A, B, np.log(L + 1e-8), levels=20, cmap='viridis')

        # Add a star at the origin (current weights)
        ax.plot(0, 0, 'r*', markersize=10, label='Current Weights')

        title = f"Loss Landscape | Step {step}"
        if grokking_step is not None:
            status = "Grokked" if step >= grokking_step and grokking_step > 0 else "Pre-Grokking"
            title += f" | {status}"

        ax.set_title(title)
        ax.set_xlabel("Random Direction 1")
        ax.set_ylabel("Random Direction 2")

        return ax

    print(f"Generating animation with {len(checkpoints)} frames...")
    ani = animation.FuncAnimation(fig, animate, frames=len(checkpoints), interval=500, blit=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ani.save(output_path, writer='ffmpeg', fps=2)
        print(f"Saved video to {output_path}")
    except Exception as e:
        print(f"Error saving mp4: {e}. Trying gif...")
        gif_path = output_path.with_suffix('.gif')
        ani.save(gif_path, writer='pillow', fps=2)
        print(f"Saved video to {gif_path}")

    plt.close()

if __name__ == "__main__":
    main()
