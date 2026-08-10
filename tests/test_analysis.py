import pytest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ModularArithmeticTransformer
from src.train import compute_fourier_concentration

def test_fourier_concentration():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64, n_layers=1)
    conc = compute_fourier_concentration(model)
    assert isinstance(conc, float)
    assert 0.0 <= conc <= 1.0

def test_embedding_rank():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64, n_layers=1)
    rank = model.get_embedding_rank()
    assert isinstance(rank, float)
    assert rank > 0.0
