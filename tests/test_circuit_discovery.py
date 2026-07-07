import torch
import pytest
from src.model import ModularArithmeticTransformer
from src.circuit_discovery import ablate_head, compute_activation_patches

@pytest.fixture
def model():
    return ModularArithmeticTransformer(
        prime=5,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1
    )

def test_ablate_head_zero(model):
    layer_idx = 0
    head_idx = 0

    # Get original weight
    layer = model.transformer.layers[layer_idx].self_attn
    out_proj = layer.out_proj
    orig_weight = out_proj.weight.data.clone()

    d_model = out_proj.weight.shape[0]
    n_heads = layer.num_heads
    head_dim = d_model // n_heads

    # Perform zero ablation
    cleanup = ablate_head(model, layer_idx, head_idx, ablation_type="zero")

    # Check that the specific head's columns are zeroed out
    start_idx = head_idx * head_dim
    end_idx = start_idx + head_dim

    assert torch.all(out_proj.weight.data[:, start_idx:end_idx] == 0)
    # Check that the other head is NOT zeroed out
    assert not torch.all(out_proj.weight.data[:, end_idx:] == 0)

    # Restore and check
    cleanup()
    assert torch.allclose(out_proj.weight.data, orig_weight)

def test_ablate_head_mean(model):
    layer_idx = 0
    head_idx = 0

    # Get original weight
    layer = model.transformer.layers[layer_idx].self_attn
    out_proj = layer.out_proj
    orig_weight = out_proj.weight.data.clone()

    d_model = out_proj.weight.shape[0]
    n_heads = layer.num_heads
    head_dim = d_model // n_heads

    # Perform mean ablation
    cleanup = ablate_head(model, layer_idx, head_idx, ablation_type="mean")

    # Check that the specific head's columns are constant across the head dimension
    start_idx = head_idx * head_dim
    end_idx = start_idx + head_dim

    ablated_slice = out_proj.weight.data[:, start_idx:end_idx]

    # For each row, all elements in the slice should be identical to the mean
    for i in range(ablated_slice.shape[0]):
        assert torch.allclose(ablated_slice[i], ablated_slice[i].mean().expand(head_dim))

    # Restore and check
    cleanup()
    assert torch.allclose(out_proj.weight.data, orig_weight)
