import numpy as np
import torch
import pytest

from analysis.information import compute_mutual_information, measure_effective_information
from analysis.phase_transition import detect_phase_transition, fit_sigmoid_transition
from analysis.scaling_laws import fit_power_law, fit_scaling_laws
from analysis.dynamics import compute_hessian_trace_proxy, measure_weight_velocity
from analysis.predict_grokking import extract_early_features, train_logistic_regression, evaluate_predictor

def test_compute_mutual_information():
    np.random.seed(42)
    # Perfectly correlated
    labels = np.array([0, 1, 0, 1, 0, 1])
    reps = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])

    mi = compute_mutual_information(reps, labels, bins=2)
    assert mi > 0

    eff_info = measure_effective_information(reps, labels)
    assert eff_info >= 0

def test_fit_sigmoid_transition():
    steps = np.arange(0, 1000, 10)
    # Create a perfect sigmoid
    x0 = 500
    k = 0.05
    accs = 1.0 / (1.0 + np.exp(-k * (steps - x0)))

    pred_k, pred_x0 = fit_sigmoid_transition(steps.tolist(), accs.tolist())

    # Should be somewhat close
    assert np.abs(pred_x0 - x0) < 50
    assert pred_k > 0

def test_fit_power_law():
    x = np.array([10, 100, 1000])
    # y = 2 * x^0.5
    y = 2.0 * np.sqrt(x)

    a, b = fit_power_law(x, y)

    assert np.isclose(a, 2.0, rtol=0.1)
    assert np.isclose(b, 0.5, rtol=0.1)

def test_measure_weight_velocity():
    state1 = {'w': torch.tensor([1.0, 2.0])}
    state2 = {'w': torch.tensor([1.0, 3.0])}
    state3 = {'w': torch.tensor([1.0, 3.0])}

    velocities = measure_weight_velocity([state1, state2, state3])

    assert len(velocities) == 2
    assert np.isclose(velocities[0], 1.0)
    assert np.isclose(velocities[1], 0.0)

def test_extract_early_features():
    history = [
        {'step': 0, 'train_loss': 1.0, 'test_loss': 1.0, 'grad_norm': 0.1},
        {'step': 100, 'train_loss': 0.8, 'test_loss': 0.9, 'grad_norm': 0.2},
        {'step': 200, 'train_loss': 0.5, 'test_loss': 0.8, 'grad_norm': 0.1}
    ]

    features = extract_early_features(history, early_steps=150)

    # Only steps 0 and 100
    assert features.shape == (3,)
    assert np.isclose(features[0], -0.2) # 0.8 - 1.0
    assert np.isclose(features[2], 0.1) # 0.9 - 0.8 (max gap)

def test_predict_grokking():
    X_train = np.array([[1.0, 0.0], [1.0, 0.1], [-1.0, 0.5], [-1.0, 0.6]])
    y_train = np.array([0, 0, 1, 1])

    model = train_logistic_regression(X_train, y_train)

    X_test = np.array([[1.0, 0.05], [-1.0, 0.55]])
    y_test = np.array([0, 1])

    acc, (ci_low, ci_high) = evaluate_predictor(model, X_test, y_test)
    assert acc == 1.0
