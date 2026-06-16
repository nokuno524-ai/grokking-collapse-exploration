import torch
import torch.nn as nn
from src.analysis.fourier_analysis import compute_fourier_concentration, get_embedding_fourier_spectrum

class MockModel(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)

def test_compute_fourier_concentration():
    # Create a perfectly periodic signal in the first dimension
    vocab_size = 59
    dim = 8

    # Generate a sine wave: low frequency should concentrate perfectly
    t = torch.arange(vocab_size, dtype=torch.float32)
    # Frequency 1 (completes 1 cycle over vocab_size)
    weights = torch.sin(2 * torch.pi * 1 * t / vocab_size).unsqueeze(1).repeat(1, dim)

    # Add a bit of noise
    weights += torch.randn_like(weights) * 0.01

    concentration = compute_fourier_concentration(weights, top_k=2)

    # Should be highly concentrated on the top frequency
    assert concentration > 0.9

def test_compute_fourier_concentration_noise():
    # Random noise should have low concentration
    vocab_size = 59
    dim = 8
    weights = torch.randn(vocab_size, dim)

    concentration = compute_fourier_concentration(weights, top_k=2)

    # Shouldn't be heavily concentrated
    assert concentration < 0.5

def test_get_embedding_fourier_spectrum():
    model = MockModel(59, 16)
    spectrum = get_embedding_fourier_spectrum(model)

    assert spectrum is not None
    assert spectrum.shape == (59, 16)
    assert not torch.isnan(spectrum).any()
