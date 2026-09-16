import torch
from src.model import ModularArithmeticTransformer

def test_model_forward():
    model = ModularArithmeticTransformer(d_model=16, n_heads=2)
    x = torch.randint(0, 59, (2, 2))
    out = model(x)
    assert out.shape == (2, 59)
