import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_weight_norm_trajectory(models):
    """
    Track L2 norm of all weight matrices across training checkpoints.

    Args:
        models: List of ModularArithmeticTransformer models (ordered by step)

    Returns:
        norms: List of float weight norms
    """
    norms = []
    for model in models:
        # Re-use the existing model.get_weight_norm() method
        norms.append(model.get_weight_norm())
    return norms


def compute_singular_value_spectrum(weight_matrix):
    """
    Compute SVD analysis of a weight matrix to detect rank collapse.

    Args:
        weight_matrix: 2D Tensor

    Returns:
        singular_values: 1D Tensor of singular values
    """
    with torch.no_grad():
        # Ensure it's a 2D matrix
        assert len(weight_matrix.shape) == 2
        # Compute SVD (we only need the singular values)
        s = torch.linalg.svdvals(weight_matrix)
    return s


def effective_rank_analysis(weight_matrix):
    """
    Track effective dimensionality of weights using Shannon entropy of the
    normalized singular value distribution.

    Args:
        weight_matrix: 2D Tensor

    Returns:
        effective_rank: float
    """
    with torch.no_grad():
        s = compute_singular_value_spectrum(weight_matrix)
        # Normalize singular values to form a probability distribution
        s_norm = s / s.sum()
        # Compute Shannon entropy
        entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
        # Effective rank is exponentiated entropy
        effective_rank = torch.exp(entropy).item()

    return effective_rank


def plot_weight_norm_trajectory(steps, norms, output_path="weight_norm_trajectory.pdf"):
    """Plot the weight norm trajectory over training steps."""
    plt.figure(figsize=(8, 5))
    plt.plot(steps, norms, linewidth=2, color='blue')
    plt.title("Weight Norm Trajectory")
    plt.xlabel("Training Step")
    plt.ylabel("Total L2 Norm")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_singular_value_spectrum(singular_values, output_path="singular_value_spectrum.pdf"):
    """Plot the singular value spectrum."""
    plt.figure(figsize=(8, 5))
    # Plot normalized singular values
    s_numpy = singular_values.numpy()
    s_norm = s_numpy / s_numpy.sum()

    plt.plot(range(1, len(s_norm) + 1), s_norm, marker='o', linestyle='-', markersize=4)
    plt.title("Singular Value Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Normalized Singular Value")
    plt.grid(True, alpha=0.3)
    # Use log scale for y-axis if the spectrum drops quickly
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
