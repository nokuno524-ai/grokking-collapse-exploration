import pytest
import torch
import torch.nn as nn
import numpy as np
from src.analysis.weight_analysis import (
    get_weight_norms,
    get_effective_rank,
    get_layer_effective_ranks,
    get_singular_value_distribution,
    calculate_weight_velocity
)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

        # Initialize with known values for testing
        nn.init.constant_(self.fc1.weight, 1.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.weight, 2.0)
        nn.init.constant_(self.fc2.bias, 0.0)

def test_get_weight_norms():
    model = SimpleModel()
    norms = get_weight_norms(model)

    # fc1.weight is 5x10 of 1.0s -> sum of squares is 50 -> norm is sqrt(50) ~ 7.07
    expected_fc1_norm = np.sqrt(50)
    # fc2.weight is 2x5 of 2.0s -> sum of squares is 10 * 4 = 40 -> norm is sqrt(40) ~ 6.32
    expected_fc2_norm = np.sqrt(40)

    assert 'fc1.weight' in norms
    assert 'fc2.weight' in norms
    assert 'total' in norms

    assert np.isclose(norms['fc1.weight'], expected_fc1_norm)
    assert np.isclose(norms['fc2.weight'], expected_fc2_norm)

    expected_total = np.sqrt(50 + 40)
    assert np.isclose(norms['total'], expected_total)

def test_get_effective_rank():
    # Rank 1 matrix (all elements same)
    tensor = torch.ones(10, 10)
    rank = get_effective_rank(tensor)
    # One singular value is ~10, rest are 0. Entropy should be close to 0 -> rank ~1.
    assert np.isclose(rank, 1.0, atol=1e-1)

    # Identity matrix (full rank)
    tensor_id = torch.eye(10)
    rank_id = get_effective_rank(tensor_id)
    # 10 equal singular values. p = 1/10. Entropy = -10 * (1/10 * log(1/10)) = log(10)
    # Rank = exp(log(10)) = 10
    assert np.isclose(rank_id, 10.0, atol=1e-1)

    # 1D tensor should return 0
    tensor_1d = torch.ones(10)
    assert get_effective_rank(tensor_1d) == 0.0

def test_get_layer_effective_ranks():
    model = SimpleModel()
    ranks = get_layer_effective_ranks(model)

    assert 'fc1.weight' in ranks
    assert 'fc2.weight' in ranks
    assert 'fc1.bias' not in ranks # Bias is 1D

    # Since weights are constant, they are rank 1
    assert np.isclose(ranks['fc1.weight'], 1.0, atol=1e-1)

def test_get_singular_value_distribution():
    tensor_id = torch.eye(5)
    s = get_singular_value_distribution(tensor_id)

    assert isinstance(s, np.ndarray)
    assert len(s) == 5
    assert np.allclose(s, 1.0)

    # 1D tensor
    assert len(get_singular_value_distribution(torch.ones(5))) == 0

def test_calculate_weight_velocity():
    model1 = SimpleModel()
    model2 = SimpleModel()

    # Zero velocity initially
    vel = calculate_weight_velocity(model1, model2)
    assert vel['total'] == 0.0

    # Change model2
    with torch.no_grad():
        model2.fc1.weight.add_(1.0) # Now it's 2.0. Diff is 1.0 everywhere (50 elements)

    vel = calculate_weight_velocity(model1, model2)
    expected_fc1_diff = np.sqrt(50 * 1.0**2)

    assert np.isclose(vel['fc1.weight'], expected_fc1_diff)
    assert vel['fc2.weight'] == 0.0
    assert np.isclose(vel['total'], expected_fc1_diff)
