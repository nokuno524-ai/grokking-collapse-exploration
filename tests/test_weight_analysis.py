import torch
import torch.nn as nn
import numpy as np
import pytest
import os

from analysis.weight_evolution import compute_layer_norms, track_norm_trajectories, track_norm_distributions, check_norm_reduction_predictor
from analysis.hessian import power_iteration, compute_top_k_eigenvalues, estimate_hessian_rank
from analysis.loss_landscape import interpolate_1d, filter_normalize, plot_2d_landscape
from analysis.weight_similarity import compute_cka, compute_weight_cka, track_similarity_trajectory

# --- Dummy Models and Data ---
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def get_dummy_data():
    inputs = torch.randn(16, 10)
    targets = torch.randn(16, 2)
    return inputs, targets

def dummy_loss(model, inputs, targets):
    outputs = model(inputs)
    return nn.MSELoss()(outputs, targets)

@pytest.fixture
def dummy_checkpoints():
    checkpoints = []
    for i in range(3):
        model = DummyModel()
        # Scale weights to simulate training/decay
        for param in model.parameters():
            param.data *= (1.0 - 0.1 * i)
        checkpoints.append({'model_state': model.state_dict(), 'step': i})
    return checkpoints

# --- Tests for Weight Evolution ---
def test_compute_layer_norms(dummy_checkpoints):
    state = dummy_checkpoints[0]['model_state']
    norms = compute_layer_norms(state)
    assert 'fc1.weight' in norms
    assert 'fc2.weight' in norms
    assert isinstance(norms['fc1.weight'], float)
    assert norms['fc1.weight'] > 0

def test_track_norm_trajectories(dummy_checkpoints):
    trajectories = track_norm_trajectories(dummy_checkpoints)
    assert len(trajectories['fc1.weight']) == 3
    # Check if norm decreases as we scaled weights down
    assert trajectories['fc1.weight'][0] > trajectories['fc1.weight'][2]

def test_check_norm_reduction_predictor():
    grokked = [10.0, 11.0, 9.5]
    collapsed = [5.0, 4.5, 5.5]
    stat, p_val, conclusion = check_norm_reduction_predictor(grokked, collapsed)
    assert isinstance(stat, float)
    assert isinstance(p_val, float)
    assert "reduction" in conclusion or "difference" in conclusion

# --- Tests for Hessian ---
def test_power_iteration():
    model = DummyModel()
    data = get_dummy_data()
    eig_val, eig_vec = power_iteration(model, dummy_loss, data, num_iterations=5)
    assert isinstance(eig_val, float)
    assert isinstance(eig_vec, torch.Tensor)
    assert eig_vec.dim() == 1

def test_compute_top_k_eigenvalues():
    model = DummyModel()
    data = get_dummy_data()
    eig_vals, eig_vecs = compute_top_k_eigenvalues(model, dummy_loss, data, k=2, num_iterations=5)
    assert len(eig_vals) == 2
    assert len(eig_vecs) == 2

def test_estimate_hessian_rank():
    eig_vals = [10.0, 5.0, 0.0001, -0.0001]
    rank = estimate_hessian_rank(eig_vals, threshold=1e-3)
    assert rank == 2

# --- Tests for Loss Landscape ---
def test_interpolate_1d(dummy_checkpoints):
    model = DummyModel()
    data = get_dummy_data()
    state_a = dummy_checkpoints[0]['model_state']
    state_b = dummy_checkpoints[1]['model_state']

    alphas, losses = interpolate_1d(model, dummy_loss, data, state_a, state_b, steps=5)
    assert len(alphas) == 5
    assert len(losses) == 5

def test_filter_normalize(dummy_checkpoints):
    state = dummy_checkpoints[0]['model_state']
    direction = {k: torch.randn_like(v) for k, v in state.items()}
    norm_dir = filter_normalize(direction, state)

    # Check that norm is scaled correctly for a layer
    w = state['fc1.weight']
    d = norm_dir['fc1.weight']

    w_norm = torch.norm(w.view(w.size(0), -1), dim=1)
    d_norm = torch.norm(d.view(d.size(0), -1), dim=1)

    # Tolerant comparison due to floating point
    assert torch.allclose(w_norm, d_norm, rtol=1e-4)

# --- Tests for Weight Similarity ---
def test_compute_cka():
    # Identical matrices should have CKA of 1.0
    a = torch.randn(10, 5)
    cka_score = compute_cka(a, a)
    assert np.isclose(cka_score, 1.0, atol=1e-5)

    # Orthogonal matrices should have CKA near 0
    b = torch.randn(10, 5)
    cka_score2 = compute_cka(a, b)
    assert cka_score2 >= 0.0 and cka_score2 <= 1.0

def test_compute_weight_cka(dummy_checkpoints):
    state_a = dummy_checkpoints[0]['model_state']
    state_b = dummy_checkpoints[1]['model_state']

    sims = compute_weight_cka(state_a, state_b)
    assert 'fc1.weight' in sims
    assert 'fc2.weight' in sims
    assert sims['fc1.weight'] <= 1.0
