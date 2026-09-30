import pytest
import numpy as np
import math
from src.rigorous_stats.stats_module import fit_power_law, fit_log_linear, bootstrap_ci, fit_logistic

def test_fit_power_law_exact():
    # y = 3 * x^2 => log y = log 3 + 2 * log x
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 3.0 * (x ** 2)
    C, b, r2 = fit_power_law(x, y)
    assert math.isclose(C, 3.0, rel_tol=1e-5)
    assert math.isclose(b, 2.0, rel_tol=1e-5)
    assert math.isclose(r2, 1.0, rel_tol=1e-5)

def test_fit_log_linear_exact():
    # log y = 0.5 + 1.5 * x => y = exp(0.5 + 1.5 * x)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.exp(0.5 + 1.5 * x)
    a, b, r2 = fit_log_linear(x, y)
    assert math.isclose(a, 0.5, rel_tol=1e-5)
    assert math.isclose(b, 1.5, rel_tol=1e-5)
    assert math.isclose(r2, 1.0, rel_tol=1e-5)

def test_fit_degenerate_data():
    x = np.array([1.0])
    y = np.array([2.0])
    # Not enough points
    C, b, r2 = fit_power_law(x, y)
    assert math.isnan(C) and math.isnan(b) and math.isnan(r2)

    a, b, r2 = fit_log_linear(x, y)
    assert math.isnan(a) and math.isnan(b) and math.isnan(r2)

    # Non-positive values for log - should filter and still fit if possible
    x = np.array([0.0, -1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    C, b, r2 = fit_power_law(x, y)
    # y = c * x^b => log y = log c + b * log x
    # For (2, 3) and (3, 4):
    # log 3 = log c + b * log 2
    # log 4 = log c + b * log 3
    # b = (log 4 - log 3) / (log 3 - log 2) = log(4/3) / log(3/2) ≈ 0.287 / 0.405 ≈ 0.709
    # c = exp(log 3 - b * log 2) = 3 / (2^b) ≈ 3 / 1.635 ≈ 1.834
    assert not math.isnan(C) and not math.isnan(b)

def test_bootstrap_ci():
    # True relationship: y = 2 * x^1.5 + noise
    rng = np.random.RandomState(42)
    x_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    seeds = [1, 2, 3, 4, 5]
    per_seed = {}
    for x in x_vals:
        for s in seeds:
            # log y = log(2) + 1.5 * log(x) + noise
            log_y = np.log(2) + 1.5 * np.log(x) + rng.normal(0, 0.1)
            per_seed[(x, s)] = np.exp(log_y)

    lo, hi = bootstrap_ci(per_seed, fit_power_law, n_boot=100, rng_seed=42)
    # Exponent should be around 1.5
    assert 1.2 < lo < 1.6
    assert 1.4 < hi < 1.8
    assert lo <= hi

def test_fit_logistic_perfect_separation():
    # If all ys are identical, it should return NaN/1.0
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0, 0, 0])
    a, b, r2 = fit_logistic(x, y)
    assert math.isnan(a) and math.isnan(b) and math.isclose(r2, 1.0)

    y2 = np.array([1, 1, 1])
    a2, b2, r22 = fit_logistic(x, y2)
    assert math.isnan(a2) and math.isnan(b2) and math.isclose(r22, 1.0)

def test_fit_logistic_normal():
    # Simple case
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    y = np.array([0, 0, 1, 1, 1])
    a, b, r2 = fit_logistic(x, y)
    assert not math.isnan(a)
    assert not math.isnan(b)
    assert b > 0  # Slope should be positive since y increases with x
    assert 0 <= r2 <= 1.0

def test_bootstrap_logistic_ci():
    x = np.array([-10.0, -5.0, -1.0, 1.0, 5.0, 10.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    # Should get a positive CI for slope
    from src.rigorous_stats.stats_module import bootstrap_logistic_ci
    lo, hi = bootstrap_logistic_ci(x, y, n_boot=50, rng_seed=42)
    assert not math.isnan(lo)
    assert not math.isnan(hi)
    assert lo <= hi
