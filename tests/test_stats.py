import numpy as np
from src.analysis.grok_detector.stats import (
    wilson_ci,
    kaplan_meier,
    cohen_d,
    bootstrap_effect_size,
    holm_mann_whitney
)

def test_wilson_ci():
    p, lower, upper = wilson_ci(5, 5)
    assert p == 1.0
    assert lower > 0.0 # roughly 0.54 with 95% conf
    assert upper == 1.0

    p, lower, upper = wilson_ci(0, 5)
    assert p == 0.0
    assert lower == 0.0
    assert upper < 1.0

def test_kaplan_meier():
    times = np.array([10, 20, 30, 40, 50])
    events = np.array([1, 0, 1, 1, 0]) # 0 means censored (failed to grok by time T)

    t, survival = kaplan_meier(times, events)
    assert len(t) == 5
    assert survival[0] < 1.0 # drop at 10
    assert survival[1] == survival[0] # censored at 20, no drop

def test_cohen_d():
    g1 = np.array([10, 12, 11, 10, 12])
    g2 = np.array([20, 22, 21, 20, 22])

    d = cohen_d(g1, g2)
    assert d < -2.0 # large negative effect

    assert cohen_d(g1, g1) == 0.0

def test_bootstrap_effect_size():
    g1 = np.array([10, 12, 11, 10, 12])
    g2 = np.array([20, 22, 21, 20, 22])

    d, lower, upper = bootstrap_effect_size(g1, g2, n_bootstraps=50)
    assert lower <= d <= upper

def test_holm_mann_whitney():
    g1 = np.array([1, 2, 3])
    g2 = np.array([10, 11, 12])

    p = holm_mann_whitney(g1, g2)
    assert p <= 0.1

    # identical groups
    p_ident = holm_mann_whitney(np.array([1, 1, 1]), np.array([1, 1, 1]))
    assert p_ident == 1.0
