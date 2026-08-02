import torch
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def compute_2d_fourier_transform(W: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D Fourier transform magnitude of a weight matrix.

    Args:
        W (torch.Tensor): A 2D weight matrix.

    Returns:
        torch.Tensor: The 2D magnitude spectrum (same shape as W).
    """
    if W.dim() != 2:
        raise ValueError("Expected a 2D weight matrix")

    # Compute 2D FFT, normalize to keep energy scale comparable
    spectrum = torch.fft.fft2(W, norm='ortho').abs()
    return spectrum

def compute_fourier_concentration(spectrum: torch.Tensor, top_k: int = 5) -> float:
    """
    Compute the fraction of spectral energy in the top-k frequencies,
    excluding the DC component (at index 0, 0).

    Args:
        spectrum (torch.Tensor): The 2D magnitude spectrum.
        top_k (int): Number of top frequencies to consider.

    Returns:
        float: Fraction of energy in top-k frequencies.
    """
    if spectrum.dim() != 2:
        raise ValueError("Expected a 2D spectrum")

    # Flatten the spectrum
    flat_spec = spectrum.flatten()

    # Assume DC component is at (0, 0), which is index 0 in flattened array
    # We create a mask to exclude it
    mask = torch.ones_like(flat_spec, dtype=torch.bool)
    mask[0] = False

    # Filtered spectrum without DC
    filtered_spec = flat_spec[mask]

    total_energy = filtered_spec.sum().item()
    if total_energy < 1e-10:
        return 0.0

    # Get top-k
    k = min(top_k, filtered_spec.numel())
    top_energy = filtered_spec.topk(k).values.sum().item()

    return top_energy / total_energy

def plot_fourier_heatmap(spectrum: torch.Tensor, title: str, output_path: str) -> None:
    """
    Generate and save a heatmap of the 2D Fourier spectrum.

    Args:
        spectrum (torch.Tensor): The 2D magnitude spectrum.
        title (str): Title for the plot.
        output_path (str): File path to save the heatmap.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return

    plt.figure(figsize=(8, 6))

    # Use log scale for better visibility, adding small epsilon to avoid log(0)
    # Using np.log1p for log(1 + x)
    img = plt.imshow(np.log1p(spectrum.detach().cpu().numpy()), cmap='viridis', aspect='auto')
    plt.colorbar(img, label='Log Magnitude')
    plt.title(title)
    plt.xlabel('Frequency (dim 1)')
    plt.ylabel('Frequency (dim 0)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
