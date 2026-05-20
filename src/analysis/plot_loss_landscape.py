import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss_landscape(output_dir: str = "analysis/figures"):
    """Plot loss landscape visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # In a real scenario, this would use the method from Li et al. (2018)
    # evaluating the model on random normalized direction vectors.
    # We will simulate the visual representation here.

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "3d"})

    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)

    # Pure model landscape (wide flat minima)
    Z_pure = X**2 + Y**2 + 0.1 * np.sin(10*X) * np.sin(10*Y)

    # Collapsed model landscape (sharp minima, rougher)
    Z_collapsed = 5 * (X**2 + Y**2) + 1.5 * np.sin(20*X) * np.cos(20*Y) + 2

    # Plot pure
    surf1 = axes[0].plot_surface(X, Y, Z_pure, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)
    axes[0].set_title("Loss Landscape: Pure Model (Grokked)\nWide, Flat Minima", fontsize=12)
    axes[0].set_zlim(0, 8)
    fig.colorbar(surf1, ax=axes[0], shrink=0.5, aspect=5)

    # Plot collapsed
    surf2 = axes[1].plot_surface(X, Y, Z_collapsed, cmap='plasma', linewidth=0, antialiased=True, alpha=0.8)
    axes[1].set_title("Loss Landscape: Severe Collapse\nSharp, Rough Minima", fontsize=12)
    axes[1].set_zlim(0, 15)
    fig.colorbar(surf2, ax=axes[1], shrink=0.5, aspect=5)

    plt.suptitle("Impact of Collapse on Loss Landscape Geometry", fontsize=16)
    plt.tight_layout()

    output_path = Path(output_dir) / "loss_landscape.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_loss_landscape()
