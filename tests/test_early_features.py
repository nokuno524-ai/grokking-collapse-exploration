import numpy as np
from eval.early_features import compute_rolling_features

def test_compute_rolling_features_empty():
    assert compute_rolling_features([]) == {}

def test_compute_rolling_features_flat():
    history = [
        {
            "step": i,
            "train_loss": 0.1,
            "test_loss": 0.2,
            "weight_norm": 10.0,
            "fourier_concentration": 0.5,
            "embedding_rank": 20.0,
            "test_acc": 0.9
        }
        for i in range(10)
    ]
    features = compute_rolling_features(history, window_size=3)

    assert len(features["step"]) == 10

    # Loss gap should be 0.1 everywhere
    np.testing.assert_allclose(features["loss_gap"], 0.1)

    # Effective rank should be 20.0 everywhere
    np.testing.assert_allclose(features["effective_rank"], 20.0)

    # Slopes should be 0 starting from index 2
    assert np.isnan(features["weight_norm_slope"][:2]).all()
    np.testing.assert_allclose(features["weight_norm_slope"][2:], 0.0, atol=1e-10)

    # Curvature should be 0
    assert np.isnan(features["test_acc_curvature"][:2]).all()
    np.testing.assert_allclose(features["test_acc_curvature"][2:], 0.0, atol=1e-10)

def test_compute_rolling_features_trend():
    # Construct a history where weight_norm grows exponentially
    # so log(weight_norm) is linear with slope 1
    # Fourier concentration grows linearly with slope 2
    # Test acc is quadratic, y = 3x^2 + 2x + 1, so curvature coeff = 3
    history = []
    for i in range(10):
        history.append({
            "step": i,
            "train_loss": 0.0,
            "test_loss": 0.0,
            "weight_norm": np.exp(float(i)),
            "fourier_concentration": 2.0 * i,
            "embedding_rank": 10.0,
            "test_acc": 3.0 * (i**2) + 2.0 * i + 1.0
        })

    features = compute_rolling_features(history, window_size=5)

    assert np.isnan(features["weight_norm_slope"][:4]).all()
    np.testing.assert_allclose(features["weight_norm_slope"][4:], 1.0, rtol=1e-5)

    assert np.isnan(features["test_acc_curvature"][:4]).all()
    np.testing.assert_allclose(features["test_acc_curvature"][4:], 3.0, rtol=1e-5)

def test_predict_grokking_crossing():
    from eval.predict_grokking import evaluate_predictor_crossing
    steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Baseline is exactly 1.0 until step 500
    features = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 50.0])
    # The standard deviation in the first 5 elements is 0.
    # Fallback to 1e-6 will be triggered. 50.0 is way larger than mean + 3*std
    cross_step = evaluate_predictor_crossing(features, steps, baseline_steps=500, threshold_sigma=3.0, direction="up")
    assert cross_step == 700

def test_predict_grokking_never_crossing_flat():
    from eval.predict_grokking import evaluate_predictor_crossing
    steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Flat time series
    features = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    cross_step = evaluate_predictor_crossing(features, steps, baseline_steps=500, threshold_sigma=3.0, direction="up")
    assert cross_step == -1

def test_malformed_logs_errors():
    from src.log_loader import load_results_json
    from pathlib import Path
    import pytest
    import json

    # We already have test_load_results_json_invalid testing for json.JSONDecodeError
    # and FileNotFoundError.
    pass
