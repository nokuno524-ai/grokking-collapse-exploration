import torch
from src.model import ModularArithmeticTransformer

def test_model_initialization():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)
    assert model.token_embed.weight.shape == (11, 32)
    assert model.output_head.weight.shape == (11, 32)

    # Check forward pass
    x = torch.tensor([[1, 2], [3, 4]])
    out = model(x)
    assert out.shape == (2, 11)

def test_metrics_computation():
    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2, d_ff=64)

    weight_norm = model.get_weight_norm()
    assert weight_norm > 0
    assert isinstance(weight_norm, float)

    rank = model.get_embedding_rank()
    assert rank > 0
    assert isinstance(rank, float)

    spectrum = model.get_embedding_fourier_spectrum()
    assert spectrum.shape == (11, 32)
    assert (spectrum >= 0).all()  # energy is non-negative
