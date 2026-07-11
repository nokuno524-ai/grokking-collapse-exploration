import pytest
import torch
import numpy as np

# Adjust imports according to actual structure
from src.representational_analysis.geometry import (
    compute_rsm, rsm_similarity, compute_participation_ratio,
    compute_mle_id, compute_neural_anisotropy, compute_cka
)
from src.representational_analysis.phase_transition import (
    detect_change_point, compute_derivative_metrics, piecewise_linear_regression
)
from src.representational_analysis.circuit import detect_circuit_emergence

def test_rsm_computation():
    # Simple orthogonal vectors
    act = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0]
    ])

    rsm = compute_rsm(act, metric='cosine')

    # Check shape
    assert rsm.shape == (3, 3)

    # Check values (cosine similarity)
    assert torch.allclose(rsm[0, 0], torch.tensor(1.0))
    assert torch.allclose(rsm[0, 1], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(rsm[0, 2], torch.tensor(-1.0), atol=1e-6)

def test_intrinsic_dimensionality():
    # 2D data embedded in 10D space
    n_samples = 100
    act = torch.zeros((n_samples, 10))
    # Fill first two dimensions with random data
    act[:, 0] = torch.randn(n_samples) * 5
    act[:, 1] = torch.randn(n_samples) * 5

    pr = compute_participation_ratio(act)
    mle_id = compute_mle_id(act, k=5)

    # Should be approximately 2
    assert 1.5 < pr < 2.5
    assert 1.0 < mle_id < 3.0

def test_cka():
    # Identical representations should have CKA = 1
    act1 = torch.randn(50, 20)
    act2 = act1.clone()

    cka_val = compute_cka(act1, act2)
    assert torch.allclose(torch.tensor(cka_val), torch.tensor(1.0), atol=1e-5)

    # Orthogonal representations
    act3 = torch.randn(50, 20)
    # Ensure they are somewhat different
    cka_diff = compute_cka(act1, act3)
    assert cka_diff < 0.99

def test_head_importance_mock():
    # We mock the importance history for detect_circuit_emergence
    # Sudden drop in differences indicates stability
    history = [
        torch.tensor([0.1]),
        torch.tensor([0.9]), # diff 0.8
        torch.tensor([0.8]), # diff 0.1
        torch.tensor([0.81]),# diff 0.01 (stable)
        torch.tensor([0.805]),# diff 0.005 (stable)
        torch.tensor([0.81]) # diff 0.005 (stable)
    ]

    # With threshold 0.05 and sustained=3, we should detect stability
    emergence_step = detect_circuit_emergence(history, threshold=0.05)

    # Diff sequence: 0.8, 0.1, 0.01, 0.005, 0.005
    # Window [0.01, 0.005, 0.005] starts at index 2 of diffs
    # Step index = 2 + 1 = 3
    assert emergence_step == 3

def test_phase_transition_detection():
    # Create synthetic grokking curve
    steps = np.arange(100)
    acc = np.zeros(100)
    # Slow climb
    acc[:40] = np.linspace(0, 0.1, 40)
    # Grokking phase
    acc[40:50] = np.linspace(0.1, 0.9, 10)
    # Plateau
    acc[50:] = 1.0

    # Test change point
    cp = detect_change_point(acc, window_size=5)
    assert 35 <= cp <= 45 # Should be around step 40

    # Test derivative metrics
    derivs = compute_derivative_metrics(acc, steps)
    velocity = derivs['velocity']
    assert velocity.shape == (100,)
    # Velocity should peak during the grokking phase
    max_vel_idx = np.argmax(velocity)
    assert 40 <= max_vel_idx <= 50

    # Test piecewise linear regression
    # Use log loss-like curve
    loss = np.ones(100)
    loss[:60] = np.linspace(2.0, 1.8, 60)
    loss[60:] = np.linspace(1.8, 0.1, 40)

    cp_reg, params = piecewise_linear_regression(steps, loss)
    assert 50 <= cp_reg <= 70
    assert params['r_squared'] > 0.8
