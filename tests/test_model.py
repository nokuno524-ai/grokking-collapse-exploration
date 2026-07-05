import pytest
import torch
from src.model import ModularArithmeticTransformer

def test_model_architecture():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64, n_layers=1)

    # 2 tokens input
    x = torch.tensor([[5, 7], [1, 2]])
    out = model(x)

    # Check output shape (batch_size, prime)
    assert out.shape == (2, 11)

def test_weight_norm():
    model = ModularArithmeticTransformer()
    norm = model.get_weight_norm()
    assert norm > 0
    assert isinstance(norm, float)

def test_fourier_spectrum():
    model = ModularArithmeticTransformer(prime=11, d_model=32)
    spectrum = model.get_embedding_fourier_spectrum()
    assert spectrum.shape == (11, 32)
    # Ensure it's using squared magnitude (energy)
    W = model.token_embed.weight.detach()
    expected = torch.fft.fft(W, dim=0).abs() ** 2
    assert torch.allclose(spectrum, expected)

def test_embedding_rank():
    model = ModularArithmeticTransformer()
    rank = model.get_embedding_rank()
    assert rank > 0
    assert isinstance(rank, float)
