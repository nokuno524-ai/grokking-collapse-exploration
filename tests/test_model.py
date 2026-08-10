import pytest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ModularArithmeticTransformer

def test_model_initialization():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64, n_layers=1)
    assert model.token_embed.weight.shape == (11, 32)
    assert model.output_head.weight.shape == (11, 32)

def test_model_forward():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64, n_layers=1)
    x = torch.randint(0, 11, (4, 2))
    out = model(x)
    assert out.shape == (4, 11)

def test_weight_norm():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64, n_layers=1)
    norm = model.get_weight_norm()
    assert isinstance(norm, float)
    assert norm > 0
