import torch
import pytest
from src.model import ModularArithmeticTransformer

def test_modular_arithmetic_transformer_forward():
    prime = 59
    d_model = 128
    batch_size = 4

    model = ModularArithmeticTransformer(
        prime=prime, d_model=d_model, n_heads=4, d_ff=512, n_layers=1
    )

    x = torch.randint(0, prime, (batch_size, 2))
    out = model(x)
    assert out.shape == (batch_size, prime)

def test_model_metrics():
    model = ModularArithmeticTransformer(prime=59, d_model=32)

    norm = model.get_weight_norm()
    assert isinstance(norm, float)
    assert norm > 0.0

    spectrum = model.get_embedding_fourier_spectrum()
    assert isinstance(spectrum, torch.Tensor)
    assert spectrum.shape == (59, 32)

    rank = model.get_embedding_rank()
    assert isinstance(rank, float)
    assert rank > 0.0
