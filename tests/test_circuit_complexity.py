import torch
import pytest
from src.model import ModularArithmeticTransformer
from src.analysis.circuit_complexity import (
    compute_attention_rank,
    compute_participation_ratio,
    compute_information_flow
)

def test_compute_attention_rank():
    model = ModularArithmeticTransformer(d_model=32, n_heads=2)
    inputs = torch.randint(0, 59, (4, 2))

    ranks = compute_attention_rank(model, inputs)
    assert "layer0_head0_rank" in ranks
    assert "layer0_head1_rank" in ranks
    assert ranks["layer0_head0_rank"] > 0

def test_compute_participation_ratio():
    # Random tensor
    activations = torch.randn(10, 32)
    pr = compute_participation_ratio(activations)
    assert pr > 0
    assert not torch.isnan(torch.tensor(pr))

def test_compute_information_flow():
    model = ModularArithmeticTransformer(prime=59, d_model=32, n_heads=2)
    inputs = torch.randint(0, 59, (4, 2))
    targets = torch.randint(0, 59, (4,))

    metrics = compute_information_flow(model, inputs, targets)
    assert "cka_input_output_layer" in metrics
    assert "pr_input" in metrics
    assert "pr_output" in metrics
