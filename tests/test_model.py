import torch
from src.model import ModularArithmeticTransformer, count_parameters

def test_model_forward():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)
    x = torch.randint(0, 59, (8, 2))
    out = model(x)
    assert out.shape == (8, 59)

def test_get_weight_norm():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)
    norm = model.get_weight_norm()
    assert isinstance(norm, float)
    assert norm > 0.0

def test_get_embedding_fourier_spectrum():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)
    spectrum = model.get_embedding_fourier_spectrum()
    assert spectrum.shape == (59, 32)
    assert spectrum.dtype == torch.float32

def test_get_embedding_rank():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)
    rank = model.get_embedding_rank()
    assert isinstance(rank, float)
    assert rank > 0.0

def test_count_parameters():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2, d_ff=64)
    params = count_parameters(model)
    assert isinstance(params, int)
    assert params > 0
