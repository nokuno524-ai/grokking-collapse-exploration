"""
Fourier analysis of learned representations for modular arithmetic models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

def extract_fourier_basis(weights: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the Fourier basis from token embeddings.

    Args:
        weights: Embedding weights tensor of shape (prime, d_model)
        top_k: Number of top frequencies to return

    Returns:
        Tuple of (top_frequencies, spectrum)
        top_frequencies: tensor of shape (top_k,)
        spectrum: the full average spectrum of shape (prime,)
    """
    # DFT along the token dimension
    spectrum_full = torch.fft.fft(weights, dim=0).abs()
    # Average across embedding dimensions
    avg_spectrum = spectrum_full.mean(dim=1)

    # Exclude DC component
    avg_spectrum_no_dc = avg_spectrum[1:]

    if avg_spectrum_no_dc.sum() < 1e-10:
        return torch.zeros(top_k, dtype=torch.long), avg_spectrum

    # Get top frequencies (note: +1 because we excluded DC)
    top_indices = avg_spectrum_no_dc.topk(min(top_k, len(avg_spectrum_no_dc))).indices + 1

    return top_indices, avg_spectrum

def track_fourier_components(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Track which Fourier components are learned in the model's embeddings.

    Args:
        model: The ModularArithmeticTransformer model

    Returns:
        Dictionary mapping embedding names to their top frequencies
    """
    results = {}
    if hasattr(model, 'token_embed'):
        top_freqs, _ = extract_fourier_basis(model.token_embed.weight.detach())
        results['token_embed'] = top_freqs

    if hasattr(model, 'pos_embed'):
        # pos_embed is only size 2, so DFT isn't very meaningful for high primes,
        # but included for completeness if needed
        top_freqs, _ = extract_fourier_basis(model.pos_embed.weight.detach(), top_k=1)
        results['pos_embed'] = top_freqs

    return results

def compare_fourier_spectra(grokked_spectrum: torch.Tensor, collapsed_spectrum: torch.Tensor) -> float:
    """
    Compare the Fourier spectrum between grokking and collapsed models.
    Returns the L2 distance between normalized spectra.

    Args:
        grokked_spectrum: Spectrum from grokked model
        collapsed_spectrum: Spectrum from collapsed model

    Returns:
        float representing the distance
    """
    # Normalize to compare shape rather than magnitude
    def normalize(s):
        s_sum = s.sum()
        if s_sum < 1e-10:
            return s
        return s / s_sum

    norm_grok = normalize(grokked_spectrum)
    norm_coll = normalize(collapsed_spectrum)

    distance = torch.norm(norm_grok - norm_coll).item()
    return distance
