import torch
import pytest
import math
from src.model import ModularArithmeticTransformer, count_parameters

def test_model_forward():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    x = torch.randint(0, 11, (4, 2))  # Batch of 4, seq_len 2

    out = model(x)

    # Check output shape: (batch_size, prime)
    assert out.shape == (4, 11)

def test_get_weight_norm():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)

    norm = model.get_weight_norm()

    # Norm should be a positive float
    assert isinstance(norm, float)
    assert norm > 0

    # Calculate expected manually
    expected_sq = sum(p.norm().item() ** 2 for p in model.parameters())
    assert math.isclose(norm, math.sqrt(expected_sq), rel_tol=1e-5)

def test_get_embedding_fourier_spectrum():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)

    spectrum = model.get_embedding_fourier_spectrum()

    # Shape should be (prime, d_model)
    assert spectrum.shape == (11, 32)

    # Should be non-negative (squared magnitude)
    assert (spectrum >= 0).all()

def test_get_embedding_rank():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)

    rank = model.get_embedding_rank()

    assert isinstance(rank, float)
    assert rank > 0

def test_count_parameters():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    num_params = count_parameters(model)

    assert num_params > 0
    assert isinstance(num_params, int)
