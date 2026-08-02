import torch
import numpy as np
import scipy.stats

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def compute_weight_statistics(W: torch.Tensor, sparsity_threshold: float = 1e-4) -> dict:
    """
    Compute weight distribution statistics including kurtosis, skewness, and sparsity.

    Args:
        W (torch.Tensor): Weight matrix.
        sparsity_threshold (float): Threshold below which a weight is considered zero (sparse).

    Returns:
        dict: A dictionary containing 'kurtosis', 'skewness', and 'sparsity'.
    """
    if W.numel() == 0:
        return {'kurtosis': 0.0, 'skewness': 0.0, 'sparsity': 0.0}

    flat_w = W.detach().cpu().numpy().flatten()

    # Calculate kurtosis and skewness using scipy
    kurt = float(scipy.stats.kurtosis(flat_w, fisher=True))  # Fisher's kurtosis (normal == 0.0)
    skew = float(scipy.stats.skew(flat_w))

    # Calculate sparsity
    sparsity = float((np.abs(flat_w) < sparsity_threshold).mean())

    return {
        'kurtosis': kurt,
        'skewness': skew,
        'sparsity': sparsity
    }

def compute_effective_rank(W: torch.Tensor) -> float:
    """
    Compute effective rank of a weight matrix using normalized singular value entropy.

    Args:
        W (torch.Tensor): A 2D weight matrix.

    Returns:
        float: The effective rank.
    """
    if W.dim() != 2:
        raise ValueError("Expected a 2D weight matrix")

    W_detached = W.detach().float()

    # Compute singular values
    s = torch.linalg.svdvals(W_detached)

    # Normalize to form a probability distribution
    s_norm = s / (s.sum() + 1e-10)

    # Compute Shannon entropy of the distribution
    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()

    # Effective rank is exp(entropy)
    effective_rank = torch.exp(entropy).item()

    return effective_rank

def plot_weight_histogram(W: torch.Tensor, title: str, output_path: str) -> None:
    """
    Generate and save a histogram of the weight matrix values.

    Args:
        W (torch.Tensor): The weight matrix.
        title (str): Title for the plot.
        output_path (str): File path to save the histogram.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    plt.figure(figsize=(8, 6))

    flat_w = W.detach().cpu().numpy().flatten()

    plt.hist(flat_w, bins=50, alpha=0.75, color='blue', edgecolor='black')

    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    # Add statistics to the plot
    kurt = float(scipy.stats.kurtosis(flat_w, fisher=True))
    skew = float(scipy.stats.skew(flat_w))
    plt.annotate(f"Kurtosis: {kurt:.2f}\nSkewness: {skew:.2f}",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
