import torch
import pytest
from src.transplant.head_transplant import split_in_proj_weight, ablate_head, transplant_head, zero_ablate_component, mean_ablate_component
from src.model import ModularArithmeticTransformer

def test_split_in_proj_weight():
    d_model = 16
    in_proj = torch.randn(3 * d_model, d_model)
    q, k, v = split_in_proj_weight(in_proj, d_model)

    assert q.shape == (d_model, d_model)
    assert k.shape == (d_model, d_model)
    assert v.shape == (d_model, d_model)
    assert torch.equal(q, in_proj[:d_model])
    assert torch.equal(k, in_proj[d_model:2*d_model])
    assert torch.equal(v, in_proj[2*d_model:])

def test_ablate_head_zero():
    d_model = 16
    n_heads = 4
    d_head = d_model // n_heads
    in_proj = torch.ones(3 * d_model, d_model)

    out_proj = ablate_head(in_proj, d_model, n_heads, head_idx=1, ablation_type="zero")
    q, k, v = split_in_proj_weight(out_proj, d_model)

    for chunk in [q, k, v]:
        assert torch.equal(chunk[d_head:2*d_head], torch.zeros(d_head, d_model))
        assert torch.equal(chunk[:d_head], torch.ones(d_head, d_model))
        assert torch.equal(chunk[2*d_head:], torch.ones(2*d_head, d_model))

def test_ablate_head_mean():
    d_model = 16
    n_heads = 4
    d_head = d_model // n_heads
    in_proj = torch.arange(3 * d_model).unsqueeze(1).repeat(1, d_model).float()

    out_proj = ablate_head(in_proj, d_model, n_heads, head_idx=0, ablation_type="mean")
    q, k, v = split_in_proj_weight(out_proj, d_model)

    for chunk in [q, k, v]:
        mean_val = chunk[:d_head].mean(dim=0, keepdim=True)
        assert torch.allclose(chunk[:d_head], mean_val.expand_as(chunk[:d_head]))

def test_zero_ablate_component():
    sd = {"comp1": torch.ones(5, 5), "comp2": torch.ones(5, 5)}
    new_sd = zero_ablate_component(sd, "comp1")
    assert torch.equal(new_sd["comp1"], torch.zeros(5, 5))
    assert torch.equal(new_sd["comp2"], torch.ones(5, 5))

def test_mean_ablate_component():
    sd = {"comp1": torch.arange(10).view(2, 5).float(), "comp2": torch.ones(5, 5)}
    new_sd = mean_ablate_component(sd, "comp1")
    expected = torch.tensor([[2.5, 3.5, 4.5, 5.5, 6.5],
                             [2.5, 3.5, 4.5, 5.5, 6.5]])
    assert torch.allclose(new_sd["comp1"], expected)
    assert torch.equal(new_sd["comp2"], torch.ones(5, 5))

def test_transplant_head():
    d_model = 8
    n_heads = 2
    d_head = d_model // n_heads
    target = torch.zeros(3 * d_model, d_model)
    donor = torch.ones(3 * d_model, d_model)

    out = transplant_head(target, donor, d_model, n_heads, head_idx=1)
    q, k, v = split_in_proj_weight(out, d_model)

    for chunk in [q, k, v]:
        assert torch.equal(chunk[:d_head], torch.zeros(d_head, d_model))
        assert torch.equal(chunk[d_head:], torch.ones(d_head, d_model))

def test_edge_cases():
    # Empty components
    sd = {}
    new_sd = zero_ablate_component(sd, "missing")
    assert new_sd == {}

    # Tiny synthetic transformer config
    model = ModularArithmeticTransformer(
        prime=7,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2
    )

    sd = model.state_dict()
    in_proj = sd["transformer.layers.0.self_attn.in_proj_weight"]

    assert in_proj.shape == (24, 8)

    ablated = ablate_head(in_proj, 8, 2, 0, "zero")
    assert ablated.shape == (24, 8)

    q, k, v = split_in_proj_weight(ablated, 8)
    assert torch.sum(q[:4, :]) == 0
    assert torch.sum(k[:4, :]) == 0
    assert torch.sum(v[:4, :]) == 0
