import torch
import torch.nn as nn
from src.analysis.weight_analysis import (
    compute_weight_norms,
    compute_weight_rank,
    compute_condition_number,
    compute_gradient_norms,
    track_weight_evolution,
    detect_collapse_from_weights
)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10, bias=False)
        self.linear2 = nn.Linear(10, 1, bias=False)
        nn.init.eye_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1.0)

def test_compute_weight_norms():
    model = SimpleModel()
    norms = compute_weight_norms(model)
    assert "linear1.weight_l1" in norms
    assert "linear1.weight_l2" in norms
    assert norms["linear1.weight_l1"] > 0

def test_compute_weight_rank():
    model = SimpleModel()
    # linear1 is identity matrix, so its rank should be high (close to 10 depending on threshold)
    ranks = compute_weight_rank(model)
    assert "linear1.weight_rank" in ranks
    assert ranks["linear1.weight_rank"] > 1

def test_compute_condition_number():
    model = SimpleModel()
    cond = compute_condition_number(model)
    assert "linear1.weight_cond" in cond
    # Identity matrix has condition number 1.0
    assert abs(cond["linear1.weight_cond"] - 1.0) < 1e-5

def test_detect_collapse_from_weights():
    # Simulate rank drop
    history = {
        "ranks": [
            {"layer1": 100},
            {"layer1": 40}  # Drops by more than 50%
        ],
        "condition_numbers": []
    }
    signatures = detect_collapse_from_weights(history)
    assert signatures["collapse_detected"] is True
    assert "layer1" in signatures["reason"]

    # Simulate condition number explosion
    history = {
        "ranks": [
            {"layer1": 100},
            {"layer1": 100}
        ],
        "condition_numbers": [
            {"layer1": 1.0},
            {"layer1": 15.0} # Explodes > 10x
        ]
    }
    signatures = detect_collapse_from_weights(history)
    assert signatures["collapse_detected"] is True
    assert "layer1" in signatures["reason"]
