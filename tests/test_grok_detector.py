import numpy as np
import pytest
from src.analysis.grok_detector.detectors import (
    piecewise_constant_detector,
    logistic_detector,
    threshold_detector,
    bootstrap_ci
)
from src.analysis.grok_detector.stats import (
    kaplan_meier_median,
    aggregate_seeds,
    cohens_d,
    cliffs_delta
)

def test_piecewise_constant_detector():
    steps = np.arange(10)
    # Step at index 5
    metric = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])

    changepoint = piecewise_constant_detector(steps, metric)
    assert changepoint == 5.0

    # Too few points
    assert piecewise_constant_detector(np.array([1, 2]), np.array([0.1, 0.9])) is None

def test_logistic_detector():
    steps = np.arange(100)
    # Create a clean logistic curve centered at 50
    metric = 1.0 / (1.0 + np.exp(-0.5 * (steps - 50)))

    changepoint = logistic_detector(steps, metric)
    assert changepoint is not None
    assert 48.0 < changepoint < 52.0

    # Flat line should fail to fit meaningfully or return within bounds
    # but scipy curve_fit might return the initial guess if maxfev is reached or if it 'fits' trivially.
    # We don't really care about flat line, but we can check it doesn't crash.
    flat_metric = np.zeros(100)
    flat_res = logistic_detector(steps, flat_metric)
    assert flat_res is None or isinstance(flat_res, float)

def test_threshold_detector():
    steps = np.arange(10)
    metric = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.0])

    assert threshold_detector(steps, metric, 0.7) == 3.0
    assert threshold_detector(steps, metric, 0.9) == 5.0
    assert threshold_detector(steps, metric, 0.99) == 7.0
    assert threshold_detector(steps, metric, 1.5) is None

def test_bootstrap_ci():
    steps = np.arange(10)
    metric = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])

    est, (lower, upper) = bootstrap_ci(steps, metric, piecewise_constant_detector, n_resamples=10)
    assert est == 5.0
    # Because data is very sharp, CI should be tight around the step
    assert not np.isnan(lower)
    assert not np.isnan(upper)

def test_kaplan_meier_median():
    # 3 events at 10, 20, 30. Median should be 20
    times = np.array([10, 20, 30])
    events = np.array([1, 1, 1])
    assert kaplan_meier_median(times, events) == 20.0

    # survival track:
    # t=10, 3 at risk, prob = 1.0
    # t=20, 3 at risk, prob = 1.0
    # t=50, 1 at risk, prob drops by (1-1/1)=0. So survival goes to 0 at t=50.
    # Therefore, 50 is correct.
    times2 = np.array([10, 20, 50])
    events2 = np.array([0, 0, 1])
    assert kaplan_meier_median(times2, events2) == 50.0

def test_aggregate_seeds():
    results = [
        {"grokking_step": 100},
        {"grokking_step": 200},
        {"grokking_step": None}, # Censored
    ]

    agg = aggregate_seeds(results, max_step=1000)
    assert agg["n_seeds"] == 3
    assert agg["n_grokked"] == 2
    assert agg["grok_rate"] == pytest.approx(2/3)
    assert agg["median"] == 200.0 # 2nd out of 3 events

def test_effect_sizes():
    g1 = np.array([10, 12, 14, 16, 18])
    g2 = np.array([20, 22, 24, 26, 28])

    d = cohens_d(g1, g2)
    assert d < -1.0 # Large negative effect size since g1 < g2

    cd = cliffs_delta(g1, g2)
    assert cd == -1.0 # All elements in g1 are less than g2

    assert np.isnan(cohens_d(np.array([1]), np.array([2])))
