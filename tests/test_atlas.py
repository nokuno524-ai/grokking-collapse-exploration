import pytest
import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer
from src.transplant.atlas import zero_ablate_component, mean_ablate_component

def test_zero_ablation():
    model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)

    # Ensure it's not zero initially
    assert not torch.all(model.state_dict()["token_embed.weight"] == 0)

    zero_ablate_component(model, "token_embed")

    assert torch.all(model.state_dict()["token_embed.weight"] == 0)

def test_mean_ablation():
    model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)

    # Set to known values
    sd = model.state_dict()
    sd["token_embed.weight"].fill_(0.0)
    sd["token_embed.weight"][0] = 100.0 # Make mean > 0
    model.load_state_dict(sd)

    mean_ablate_component(model, "token_embed", dataloader=None, device=torch.device("cpu"))

    val = model.state_dict()["token_embed.weight"].mean().item()
    assert torch.allclose(model.state_dict()["token_embed.weight"], torch.tensor(val))

def test_resample_ablation():
    from src.transplant.atlas import resample_ablate_component
    model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)
    other_model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32, n_layers=2)

    other_sd = other_model.state_dict()

    assert not torch.allclose(model.state_dict()["token_embed.weight"], other_sd["token_embed.weight"])

    resample_ablate_component(model, "token_embed", other_sd)

    assert torch.allclose(model.state_dict()["token_embed.weight"], other_sd["token_embed.weight"])
