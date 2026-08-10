import numpy as np
import sys
import os

# Ensure theory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theory.framework import (
    compute_mutual_information,
    weight_norm_trajectory,
    PhaseTransitionModel,
    predict_grokking
)

def test_compute_mutual_information():
    mi_pure = compute_mutual_information(0.0, 10.0)
    mi_collapsed = compute_mutual_information(0.5, 10.0)
    mi_total_collapse = compute_mutual_information(1.0, 10.0)

    assert mi_pure > mi_collapsed
    assert mi_total_collapse == 0.0

def test_weight_norm_trajectory():
    t = np.array([0, 10, 100])
    wd = 0.1
    eta = 0.5
    initial_norm = 10.0

    w_traj = weight_norm_trajectory(t, wd, eta, initial_norm)

    assert len(w_traj) == 3
    assert w_traj[0] == initial_norm
    # With wd=0.1, eta=0.5, steady state is 0.5/0.1 = 5.0
    # initial_norm=10.0, so it should decay towards 5.0
    assert w_traj[-1] < initial_norm
    assert np.isclose(w_traj[-1], 5.0 + 5.0 * np.exp(-0.1 * 100))


def test_phase_transition():
    model = PhaseTransitionModel(critical_threshold=0.1)
    assert model.is_grokking_expected(0.0) is True
    assert model.is_grokking_expected(0.05) is True
    assert model.is_grokking_expected(0.15) is False
    assert model.is_grokking_expected(0.3) is False


def test_predict_grokking():
    assert predict_grokking(0.05, 1.0) is True
    assert predict_grokking(0.15, 1.0) is False
    # Test high wd preventing grokking
    assert predict_grokking(0.0, 3.0) is False
