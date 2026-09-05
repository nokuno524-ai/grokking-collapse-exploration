import numpy as np
import pytest
from src.analysis.grok_detector.stats import (
    detect_grokking_step,
    bootstrap_median_ci,
    kaplan_meier_survival,
    log_rank_test_multi
)

def test_detect_grokking_step_normal():
    steps = np.array([100, 200, 300, 400, 500, 600, 700])
    accs = np.array([0.5, 0.6, 0.96, 0.97, 0.98, 0.99, 1.0])
    # threshold 0.95, stability 3
    # crosses at 300, stays for 300,400,500. So detects 300.
    assert detect_grokking_step(steps, accs, threshold=0.95, stability_window=3) == 300

def test_detect_grokking_step_transient():
    steps = np.array([100, 200, 300, 400, 500, 600, 700])
    accs = np.array([0.5, 0.96, 0.90, 0.96, 0.97, 0.98, 1.0])
    # crosses at 200, but drops at 300.
    # crosses again at 400, stays for 400,500,600. So detects 400.
    assert detect_grokking_step(steps, accs, threshold=0.95, stability_window=3) == 400

def test_detect_grokking_step_never():
    steps = np.array([100, 200, 300, 400, 500])
    accs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    assert detect_grokking_step(steps, accs, threshold=0.95, stability_window=3) is None

def test_bootstrap_median_ci_normal():
    data = np.array([100, 100, 200, 300, 400, 500, 500])
    median, lci, uci = bootstrap_median_ci(data, seed=42)
    assert median == 300.0
    assert lci <= median <= uci
    assert not np.isnan(lci)

def test_bootstrap_median_ci_nans():
    data = np.array([100, 200, np.nan, 300, 400, np.nan])
    median, lci, uci = bootstrap_median_ci(data, seed=42)
    assert median == 250.0  # (200+300)/2
    assert lci <= median <= uci

def test_bootstrap_median_ci_identical():
    data = np.array([200, 200, 200, 200])
    median, lci, uci = bootstrap_median_ci(data, seed=42)
    assert median == 200.0
    assert lci == 200.0
    assert uci == 200.0

def test_bootstrap_median_ci_empty():
    data = np.array([np.nan, np.nan])
    median, lci, uci = bootstrap_median_ci(data)
    assert np.isnan(median)
    assert np.isnan(lci)

def test_kaplan_meier_survival():
    times = np.array([10, 20, 30, 40, 50])
    # 20 is censored, rest are events
    censored = np.array([False, True, False, False, False])

    utimes, surv, lci, uci = kaplan_meier_survival(times, censored)

    assert len(utimes) == 5
    # t=10, 1 event out of 5: surv = 0.8
    assert np.isclose(surv[0], 0.8)

    # t=20, 1 censored out of 4: surv = 0.8
    assert np.isclose(surv[1], 0.8)

    # t=30, 1 event out of 3: surv = 0.8 * (2/3) = 0.5333
    assert np.isclose(surv[2], 0.8 * (2.0/3.0))

def test_log_rank_test_multi():
    groups = {
        "A": (np.array([10, 20, 30]), np.array([False, False, False])),
        "B": (np.array([40, 50, 60]), np.array([False, False, False]))
    }
    p_vals = log_rank_test_multi(groups)
    assert ("A", "B") in p_vals or ("B", "A") in p_vals

    # A always happens before B, so they are significantly different
    pval = p_vals.get(("A", "B"), p_vals.get(("B", "A")))
    assert pval < 0.1
