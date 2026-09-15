import torch
import torch.nn as nn
import pytest

from src.transplant.transplant_rescue import random_basis_swap, patch_state_dict, COMPONENT_PATTERNS
from src.transplant.head_transplant import transplant_attention_head

def test_random_basis_swap():
    rng = torch.Generator().manual_seed(42)
    # 2D case
    w = torch.randn(10, 5)
    w_swap = random_basis_swap(w, rng)

    # Should have same shape
    assert w.shape == w_swap.shape

    # Should have same frobenius norm
    assert torch.allclose(torch.norm(w), torch.norm(w_swap), atol=1e-5)

    # Should have same singular values
    _, s, _ = torch.linalg.svd(w, full_matrices=False)
    _, s_swap, _ = torch.linalg.svd(w_swap, full_matrices=False)
    assert torch.allclose(s, s_swap, atol=1e-4)

    # 1D case (permutation)
    w1d = torch.randn(10)
    w1d_swap = random_basis_swap(w1d, rng)
    assert w1d.shape == w1d_swap.shape
    assert torch.allclose(torch.norm(w1d), torch.norm(w1d_swap), atol=1e-5)

def test_patch_state_dict():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(5, 5)
            self.linear2 = nn.Linear(5, 5)
            # rename for regex matching
            self.linear1_named = nn.Linear(5, 5)
            self.transformer = nn.Module()
            self.transformer.layers = nn.ModuleList([nn.Module()])
            self.transformer.layers[0].linear1 = nn.Linear(5, 5)

    model1 = DummyModel()
    model2 = DummyModel()

    # Need to manipulate so it has keys matched by COMPONENT_PATTERNS
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    # Check linear1 replacement
    patched = patch_state_dict(sd1, sd2, "linear1")
    key = "transformer.layers.0.linear1.weight"

    # Assert patched key has model2's weights
    assert torch.equal(patched[key], sd2[key])

    # Assert unpatched key retains model1's weights
    assert torch.equal(patched["linear2.weight"], sd1["linear2.weight"])

def test_transplant_attention_head():
    # Setup dummy d_model, n_heads
    d_model = 16
    n_heads = 4
    head_dim = d_model // n_heads # 4

    sd1 = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.zeros(3 * d_model, d_model),
        "transformer.layers.0.self_attn.in_proj_bias": torch.zeros(3 * d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.zeros(d_model, d_model)
    }

    sd2 = {
        "transformer.layers.0.self_attn.in_proj_weight": torch.ones(3 * d_model, d_model),
        "transformer.layers.0.self_attn.in_proj_bias": torch.ones(3 * d_model),
        "transformer.layers.0.self_attn.out_proj.weight": torch.ones(d_model, d_model)
    }

    # Transplant head 1
    patched = transplant_attention_head(
        sd1, sd2, layer_idx=0, head_idx=1, d_model=d_model, n_heads=n_heads
    )

    # Check in_proj Q, K, V (head 1 is indices 4:8)
    for offset in [0, d_model, 2 * d_model]:
        # Head 1 should be ones
        assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"][offset+4 : offset+8, :] == 1)
        # Head 0 should be zeros
        assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"][offset+0 : offset+4, :] == 0)
        # Head 2 should be zeros
        assert torch.all(patched["transformer.layers.0.self_attn.in_proj_weight"][offset+8 : offset+12, :] == 0)

    # Check out_proj (head 1 is columns 4:8)
    assert torch.all(patched["transformer.layers.0.self_attn.out_proj.weight"][:, 4:8] == 1)
    assert torch.all(patched["transformer.layers.0.self_attn.out_proj.weight"][:, 0:4] == 0)
