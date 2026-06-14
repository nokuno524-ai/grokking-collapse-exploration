"""
Fourier analysis utilities for tracking grokking mechanics.
"""

import math
from typing import Dict, List, Tuple
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def extract_fourier_basis(model: torch.nn.Module) -> torch.Tensor:
    """
    Compute 2D Fourier transform of the embedding matrix.
    Useful for detecting circular convolution structure in modular arithmetic.

    Args:
        model: ModularArithmeticTransformer instance

    Returns:
        torch.Tensor of shape (prime, d_model) representing the magnitude spectrum
    """
    W = model.token_embed.weight.detach()  # (prime, d_model)
    # FFT along the prime (token) dimension to find frequencies
    spectrum = torch.fft.fft(W, dim=0).abs()
    return spectrum

def compute_fourier_spectrum(activations: torch.Tensor) -> torch.Tensor:
    """
    Compute frequency spectrum of hidden activations across the batch/vocab dimension.

    Args:
        activations: shape (batch, ..., hidden_dim) or similar where we analyze dim=0
                     Assuming activations map to the modular domain if aligned.
                     If it's arbitrary activations, we just take FFT over dim=0.
    Returns:
        Magnitude spectrum
    """
    spectrum = torch.fft.fft(activations, dim=0).abs()
    return spectrum

def track_fourier_evolution(checkpoints: List[Tuple[int, torch.nn.Module]],
                            inputs: torch.Tensor) -> Dict[int, torch.Tensor]:
    """
    Track how Fourier structure emerges over training.

    Args:
        checkpoints: List of (step_num, model_at_step)
        inputs: Not strictly needed if we just track embedding, but can be used for activations.
                For now we just track embedding spectrum.

    Returns:
        Dict mapping step number to frequency spectrum
    """
    evolution = {}
    for step, model in checkpoints:
        evolution[step] = extract_fourier_basis(model)
    return evolution

def identify_dominant_frequencies(model: torch.nn.Module, threshold: float = 0.5) -> List[int]:
    """
    Find which frequency components carry the most signal.

    Args:
        model: The model
        threshold: The relative magnitude threshold to consider a frequency "dominant"
                   (relative to the max magnitude)

    Returns:
        List of dominant frequency indices
    """
    spectrum = extract_fourier_basis(model)
    # Average across the d_model dimension to get global frequency importance
    avg_spectrum = spectrum.mean(dim=1)

    # Normalize by max to use relative threshold
    max_val = avg_spectrum.max()
    normalized_spectrum = avg_spectrum / (max_val + 1e-9)

    dominant = []
    for freq_idx, val in enumerate(normalized_spectrum):
        if val >= threshold:
            dominant.append(freq_idx)
    return dominant

def plot_fourier_spectrum(spectrum: torch.Tensor, title: str, save_path: str = None):
    """
    Generate publication-quality heatmap of the Fourier spectrum.

    Args:
        spectrum: (prime, d_model) tensor
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))

    spectrum_np = spectrum.cpu().numpy()

    im = plt.imshow(spectrum_np, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Magnitude')
    plt.title(title)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Frequency (k)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
