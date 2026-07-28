import pytest
import torch
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ModularArithmeticTransformer

def test_model_forward():
    """Test model forward pass returns correct shape."""
    prime = 11
    batch_size = 4
    model = ModularArithmeticTransformer(prime=prime, d_model=32, n_heads=2, d_ff=64)

    # Inputs: (batch, 2)
    x = torch.randint(0, prime, (batch_size, 2))

    logits = model(x)

    # Output should be (batch, prime)
    assert logits.shape == (batch_size, prime)

def test_model_weight_norm():
    """Test weight norm calculation."""
    model = ModularArithmeticTransformer(d_model=32, n_heads=2)
    norm = model.get_weight_norm()

    assert isinstance(norm, float)
    assert norm > 0.0

def test_model_embedding_metrics():
    """Test embedding rank and fourier spectrum."""
    model = ModularArithmeticTransformer(d_model=32, n_heads=2)

    rank = model.get_embedding_rank()
    assert isinstance(rank, float)
    assert rank > 0.0

    spectrum = model.get_embedding_fourier_spectrum()
    assert spectrum.shape == (model.prime, model.d_model)
    assert spectrum.dtype == torch.float32
