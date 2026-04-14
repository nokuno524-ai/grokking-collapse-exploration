import torch
import pytest
from src.model import ModularArithmeticTransformer

def test_model_forward():
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4, d_ff=512, n_layers=1)
    x = torch.randint(0, 59, (10, 2))

    # Test without return_attn
    logits = model(x)
    assert logits.shape == (10, 59)

    # Test with return_attn
    logits, attn = model(x, return_attn=True)
    assert logits.shape == (10, 59)
    assert attn.shape == (10, 2, 2)  # batch, tgt_len, src_len

def test_model_weight_norm():
    model = ModularArithmeticTransformer()
    norm = model.get_weight_norm()
    assert isinstance(norm, float)
    assert norm > 0

def test_model_embedding_rank():
    model = ModularArithmeticTransformer()
    rank = model.get_embedding_rank()
    assert isinstance(rank, float)
    assert rank > 0

def test_model_embedding_fourier_spectrum():
    model = ModularArithmeticTransformer(prime=59, d_model=128)
    spectrum = model.get_embedding_fourier_spectrum()
    assert spectrum.shape == (59, 128)
    assert torch.all(spectrum >= 0)  # Magnitudes should be positive
