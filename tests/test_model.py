import pytest
import torch
from src.model import ModularArithmeticTransformer, count_parameters

def test_model_forward_shape():
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4, d_ff=512, n_layers=1)

    batch_size = 8
    x = torch.randint(0, 59, (batch_size, 2))

    logits = model(x)
    assert logits.shape == (batch_size, 59)

def test_model_parameter_count():
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4, d_ff=512, n_layers=1)
    # Token embed: 59 * 128 = 7552
    # Pos embed: 2 * 128 = 256
    # Transformer: ~ 128*128*3 (QKV) + 128*128 (out) + 128*512 (ff1) + 512*128 (ff2) + LN + biases
    # Head: 128 * 59 = 7552

    params = count_parameters(model)
    # The actual exact count based on our run output was 213,947
    assert params == 213947

def test_fourier_spectrum():
    model = ModularArithmeticTransformer(prime=59, d_model=128)
    spectrum = model.get_embedding_fourier_spectrum()

    assert spectrum.shape == (59, 128)
    # The spectrum from torch.fft.fft().abs() ** 2 should be non-negative
    assert torch.all(spectrum >= 0)

def test_embedding_rank():
    model = ModularArithmeticTransformer(prime=59, d_model=128)
    rank = model.get_embedding_rank()

    assert isinstance(rank, float)
    assert rank > 0.0
