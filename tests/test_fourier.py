import torch
import numpy as np
import pytest
from src.analysis.fourier import compute_2d_fourier_transform, compute_fourier_concentration

def test_compute_2d_fourier_transform():
    # Test valid 2D tensor
    W = torch.randn(10, 10)
    spectrum = compute_2d_fourier_transform(W)

    assert spectrum.shape == (10, 10)
    assert not torch.is_complex(spectrum)
    assert torch.all(spectrum >= 0)

    # Test invalid dimensions
    with pytest.raises(ValueError):
        compute_2d_fourier_transform(torch.randn(10))

    with pytest.raises(ValueError):
        compute_2d_fourier_transform(torch.randn(10, 10, 10))

def test_compute_fourier_concentration():
    # Create a mock spectrum where we know the exact values
    spectrum = torch.zeros(5, 5)

    # DC component
    spectrum[0, 0] = 100.0

    # Some other frequencies
    spectrum[1, 1] = 10.0
    spectrum[2, 2] = 5.0
    spectrum[3, 3] = 3.0
    spectrum[4, 4] = 2.0

    # Total energy excluding DC: 10 + 5 + 3 + 2 = 20

    # Top 1 excluding DC: 10
    # Concentration = 10 / 20 = 0.5
    c1 = compute_fourier_concentration(spectrum, top_k=1)
    assert np.isclose(c1, 0.5)

    # Top 2 excluding DC: 10 + 5 = 15
    # Concentration = 15 / 20 = 0.75
    c2 = compute_fourier_concentration(spectrum, top_k=2)
    assert np.isclose(c2, 0.75)

    # All non-DC: 20 / 20 = 1.0
    c_all = compute_fourier_concentration(spectrum, top_k=4)
    assert np.isclose(c_all, 1.0)

    # Test invalid dimensions
    with pytest.raises(ValueError):
        compute_fourier_concentration(torch.randn(10))
