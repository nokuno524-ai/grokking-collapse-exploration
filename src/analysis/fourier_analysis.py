"""
Mechanistic tools for Fourier analysis on model weights.
"""

import torch
import torch.nn as nn
from typing import Optional, Union


def compute_fourier_concentration(weights: torch.Tensor, top_k: int = 5) -> float:
    """
    Measure how concentrated the Fourier spectrum of a weight matrix is on the top-k frequencies.
    High concentration indicates a strong periodic pattern often associated with grokking
    in modular arithmetic tasks.

    Args:
        weights: A tensor of shape (vocab_size, dim) or similar where the first dimension
                 is the sequential or modular domain.
        top_k: Number of top frequencies to consider in the concentration metric.

    Returns:
        float: The ratio of energy in the top_k non-DC frequencies to the total energy
               in all non-DC frequencies.
    """
    # Ensure detached
    W = weights.detach().cpu()

    if len(W.shape) == 1:
        W = W.unsqueeze(1)

    # DFT along the first dimension
    spectrum = torch.fft.fft(W, dim=0).abs()

    # Average across dimensions if multidimensional
    avg_spectrum = spectrum.mean(dim=1) if spectrum.shape[1] > 1 else spectrum.squeeze(1)

    # Exclude DC component
    if len(avg_spectrum) > 1:
        avg_spectrum = avg_spectrum[1:]

    total_energy = avg_spectrum.sum()
    if total_energy < 1e-10:
        return 0.0

    # Compute top-k energy
    k = min(top_k, len(avg_spectrum))
    if k == 0:
        return 0.0

    top_energy = avg_spectrum.topk(k).values.sum()

    return (top_energy / total_energy).item()


def get_embedding_fourier_spectrum(model: nn.Module) -> Optional[torch.Tensor]:
    """
    Compute the Fourier spectrum of the token embedding matrix of a ModularArithmeticTransformer.

    Args:
        model: A trained ModularArithmeticTransformer model.

    Returns:
        torch.Tensor or None: The magnitude of the DFT of the embedding weights,
                              or None if not found.
    """
    if hasattr(model, 'token_embed') and hasattr(model.token_embed, 'weight'):
        W = model.token_embed.weight.detach()
        spectrum = torch.fft.fft(W, dim=0).abs()
        return spectrum
    return None
