import pytest
import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer, count_parameters

def test_model_initialization():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    assert model.prime == 11
    assert model.d_model == 32
    assert model.n_heads == 2

def test_count_parameters():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    num_params = count_parameters(model)
    assert num_params > 0
    assert num_params == sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_forward_pass():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    batch_size = 4
    x = torch.randint(0, 11, (batch_size, 2))
    logits = model(x)

    assert logits.shape == (batch_size, 11)

def test_get_weight_norm():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    norm = model.get_weight_norm()
    assert isinstance(norm, float)
    assert norm > 0.0

def test_get_embedding_fourier_spectrum():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    spectrum = model.get_embedding_fourier_spectrum()
    assert spectrum.shape == (11, 32)
    assert torch.all(spectrum >= 0.0)

def test_get_embedding_rank_normal():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    rank = model.get_embedding_rank()
    assert isinstance(rank, float)
    assert rank > 0.0

def test_get_embedding_rank_zero_sum():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    # Manually zero out weights to force sum of svds to be zero
    with torch.no_grad():
        model.token_embed.weight.fill_(0.0)
    rank = model.get_embedding_rank()
    assert rank == 0.0
