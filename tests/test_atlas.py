import pytest
import torch
from src.model import ModularArithmeticTransformer
from src.transplant.atlas import zero_ablate_component, mean_ablate_component, resample_ablate_component

@pytest.fixture
def base_model():
    torch.manual_seed(42)
    return ModularArithmeticTransformer(prime=59, d_model=32, n_heads=4, n_layers=2)

def test_zero_ablate(base_model):
    base_sd = base_model.state_dict()
    ablated_sd = zero_ablate_component(base_sd, 'token_embed')

    assert torch.all(ablated_sd['token_embed.weight'] == 0)
    # Check other parts unaffected
    assert not torch.all(ablated_sd['pos_embed.weight'] == 0)

def test_mean_ablate_fallback(base_model):
    base_sd = base_model.state_dict()
    ablated_sd = mean_ablate_component(base_sd, 'token_embed')

    mean_val = base_sd['token_embed.weight'].mean()
    assert torch.allclose(ablated_sd['token_embed.weight'], torch.full_like(ablated_sd['token_embed.weight'], mean_val))

def test_resample_ablate(base_model):
    base_sd = base_model.state_dict()
    ablated_sd = resample_ablate_component(base_sd, 'token_embed')

    # Distributions (sorted values) should be identical
    orig_sorted, _ = torch.sort(base_sd['token_embed.weight'].flatten())
    resampled_sorted, _ = torch.sort(ablated_sd['token_embed.weight'].flatten())
    assert torch.allclose(orig_sorted, resampled_sorted)

    # But actual values should be permuted
    assert not torch.allclose(base_sd['token_embed.weight'], ablated_sd['token_embed.weight'])
