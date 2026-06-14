import torch
import math
from src.fourier_analysis import compute_fourier_spectrum

def test_fourier_spectrum_synthetic():
    # Create synthetic data with known frequencies
    N = 59
    x = torch.arange(N).float()

    # Signal 1: frequency k=3
    k1 = 3
    signal1 = torch.sin(2 * math.pi * k1 * x / N)

    # Signal 2: frequency k=10
    k2 = 10
    signal2 = torch.cos(2 * math.pi * k2 * x / N)

    # Combine signals along d_model dimension (shape: N, 2)
    activations = torch.stack([signal1, signal2], dim=1)

    # Compute spectrum
    spectrum = compute_fourier_spectrum(activations)

    # Expected output:
    # spectrum[k1, 0] should be large (and N-k1 due to symmetry)
    # spectrum[k2, 1] should be large (and N-k2)

    # Normalize
    spec_dim0 = spectrum[:, 0] / spectrum[:, 0].max()
    spec_dim1 = spectrum[:, 1] / spectrum[:, 1].max()

    # Check peaks
    assert spec_dim0[k1].item() > 0.9
    assert spec_dim0[N - k1].item() > 0.9
    assert spec_dim1[k2].item() > 0.9
    assert spec_dim1[N - k2].item() > 0.9

    # Check non-peaks are small
    assert spec_dim0[k2].item() < 0.1
    assert spec_dim1[k1].item() < 0.1
