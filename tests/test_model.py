import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import ModularArithmeticTransformer

def test_model_forward():
    batch_size = 16
    prime = 23
    model = ModularArithmeticTransformer(prime=prime, d_model=32, n_heads=2, d_ff=64)

    # Inputs: (batch_size, 2)
    inputs = torch.randint(0, prime, (batch_size, 2))

    logits = model(inputs)

    # Output shape: (batch_size, prime)
    assert logits.shape == (batch_size, prime)

def test_model_metrics():
    model = ModularArithmeticTransformer(prime=11, d_model=16, n_heads=2, d_ff=32)

    weight_norm = model.get_weight_norm()
    assert isinstance(weight_norm, float)
    assert weight_norm > 0

    rank = model.get_embedding_rank()
    assert isinstance(rank, float)
    assert rank > 0

    spectrum = model.get_embedding_fourier_spectrum()
    assert spectrum.shape == (11, 16)
