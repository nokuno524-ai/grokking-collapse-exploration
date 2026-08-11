import pytest
import torch
from src.model import ModularArithmeticTransformer

def test_model_initialization():
    model = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2, d_ff=32, n_layers=1)
    assert model.token_embed.weight.shape == (7, 16)
    assert model.pos_embed.weight.shape == (2, 16)

def test_forward_shape():
    model = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2, d_ff=32, n_layers=1)
    x = torch.randint(0, 7, (4, 2))
    logits = model(x)
    assert logits.shape == (4, 7)

def test_weight_norm():
    model = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2, d_ff=32, n_layers=1)
    norm = model.get_weight_norm()
    assert norm > 0

def test_embedding_rank():
    model = ModularArithmeticTransformer(prime=7, d_model=16, n_heads=2, d_ff=32, n_layers=1)
    rank = model.get_embedding_rank()
    assert rank > 0
