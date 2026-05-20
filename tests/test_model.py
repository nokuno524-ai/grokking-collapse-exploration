import pytest
import torch
from src.model import ModularArithmeticTransformer, count_parameters

def test_model_initialization():
    model = ModularArithmeticTransformer(prime=7, d_model=32, n_heads=2, d_ff=64)
    assert model.prime == 7
    assert model.d_model == 32
    assert model.n_heads == 2
    assert count_parameters(model) > 0

def test_model_forward():
    model = ModularArithmeticTransformer(prime=7, d_model=32, n_heads=2, d_ff=64)
    # batch size 4, seq len 2
    x = torch.randint(0, 7, (4, 2))
    logits = model(x)
    assert logits.shape == (4, 7)

def test_model_weight_norm():
    model = ModularArithmeticTransformer(prime=7, d_model=32, n_heads=2, d_ff=64)
    norm = model.get_weight_norm()
    assert isinstance(norm, float)
    assert norm > 0

def test_model_embedding_fourier_spectrum():
    model = ModularArithmeticTransformer(prime=7, d_model=32, n_heads=2, d_ff=64)
    spectrum = model.get_embedding_fourier_spectrum()
    assert spectrum.shape == (7, 32)
    assert not torch.isnan(spectrum).any()

def test_model_embedding_rank():
    model = ModularArithmeticTransformer(prime=7, d_model=32, n_heads=2, d_ff=64)
    rank = model.get_embedding_rank()
    assert isinstance(rank, float)
    assert rank > 0
