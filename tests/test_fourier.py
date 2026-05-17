import torch
import pytest
from src.fourier_analysis import extract_fourier_basis, track_fourier_components, compare_fourier_spectra
from src.model import ModularArithmeticTransformer

def test_extract_fourier_basis():
    # Create a dummy embedding weights tensor (prime=59, d_model=128)
    prime = 59
    d_model = 128
    weights = torch.randn(prime, d_model)

    top_indices, spectrum = extract_fourier_basis(weights, top_k=5)

    assert spectrum.shape == (prime,)
    assert top_indices.shape == (5,)
    assert (top_indices > 0).all(), "DC component should be excluded"
    assert (top_indices < prime).all()

def test_track_fourier_components():
    model = ModularArithmeticTransformer(prime=59, d_model=32)
    results = track_fourier_components(model)

    assert 'token_embed' in results
    assert 'pos_embed' in results
    assert results['token_embed'].shape[0] == 5
    assert results['pos_embed'].shape[0] == 1

def test_compare_fourier_spectra():
    prime = 59
    grok_spectrum = torch.rand(prime)
    col_spectrum = torch.rand(prime)

    # Distance to self should be zero
    assert compare_fourier_spectra(grok_spectrum, grok_spectrum) < 1e-5

    # Distance between two random distributions should be non-zero
    dist = compare_fourier_spectra(grok_spectrum, col_spectrum)
    assert dist > 0.0
