import torch
from src.model import ModularArithmeticTransformer
from src.transplant.atlas import zero_ablate_component, mean_ablate_component, resample_ablate_component

def test_zero_ablation():
    model = ModularArithmeticTransformer(d_model=32, d_ff=128, n_heads=2)
    sd = model.state_dict()

    keys = ["transformer.layers.0.linear1.weight"]
    zeroed_sd = zero_ablate_component(sd, keys)

    assert torch.allclose(zeroed_sd["transformer.layers.0.linear1.weight"], torch.zeros_like(sd["transformer.layers.0.linear1.weight"]))

def test_mean_ablation():
    model = ModularArithmeticTransformer(d_model=32, d_ff=128, n_heads=2)
    sd = model.state_dict()

    keys = ["transformer.layers.0.linear1.weight"]
    mean_sd = mean_ablate_component(sd, keys)

    mean_val = sd["transformer.layers.0.linear1.weight"].mean()
    assert torch.allclose(mean_sd["transformer.layers.0.linear1.weight"], torch.full_like(sd["transformer.layers.0.linear1.weight"], mean_val))

def test_resample_ablation():
    model = ModularArithmeticTransformer(d_model=32, d_ff=128, n_heads=2)
    sd = model.state_dict()

    keys = ["transformer.layers.0.linear1.weight"]
    resampled_sd = resample_ablate_component(sd, keys, seed=42)

    # Check that it's different but has roughly same mean
    assert not torch.allclose(resampled_sd["transformer.layers.0.linear1.weight"], sd["transformer.layers.0.linear1.weight"])
    assert torch.isclose(resampled_sd["transformer.layers.0.linear1.weight"].mean(), sd["transformer.layers.0.linear1.weight"].mean(), atol=1e-1)
