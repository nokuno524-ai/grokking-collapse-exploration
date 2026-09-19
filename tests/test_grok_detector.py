import pytest
import numpy as np
import tempfile
import json
import os
from pathlib import Path

from src.analysis.grok_detector.detectors import smooth_series, max_2nd_derivative_detector
from src.analysis.grok_detector.stats import compute_median_iqr, holm_mann_whitney, permutation_spearman
from src.analysis.grok_detector.run_aggregator import aggregate_and_plot


def generate_s_curve(steps, midpoint, steepness, max_val, noise_std=0.0):
    """Generates a synthetic sigmoid S-curve."""
    steps_arr = np.array(steps)
    vals = max_val / (1 + np.exp(-steepness * (steps_arr - midpoint)))
    if noise_std > 0:
        rng = np.random.default_rng(42)
        vals += rng.normal(0, noise_std, size=len(vals))
        vals = np.clip(vals, 0, 1.0)
    return vals


def test_smooth_series():
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    # window_size = 3: [ (1+1+2)/3, (1+2+3)/3, (2+3+4)/3, (3+4+5)/3, (4+5+5)/3 ]
    # = [1.33, 2.0, 3.0, 4.0, 4.66]
    smoothed = smooth_series(series, 3)
    assert len(smoothed) == 5
    assert np.allclose(smoothed[1:4], [2.0, 3.0, 4.0])

    # window_size 1 should return identical
    assert np.allclose(smooth_series(series, 1), series)


def test_max_2nd_derivative_detector_clean_curve():
    steps = list(range(0, 5000, 50))
    # True inflection point is at step 2000
    vals = generate_s_curve(steps, 2000, 0.005, 1.0, noise_std=0.0)

    result = max_2nd_derivative_detector(steps, vals, window_size=3)

    assert result['grokked'] is True
    # The max 2nd derivative (acceleration) should happen BEFORE the midpoint (inflection).
    # Inflection point is where 2nd derivative is 0.
    # Max acceleration is around x = midpoint - 1.317/steepness
    expected_peak = 2000 - 1.317/0.005
    # Should be around 1736. The closest step is 1750.
    assert result['cliff_step'] is not None
    assert 1600 <= result['cliff_step'] <= 1800


def test_max_2nd_derivative_detector_no_cliff():
    steps = list(range(0, 5000, 50))
    # Never goes above 0.4
    vals = generate_s_curve(steps, 2000, 0.005, 0.4, noise_std=0.0)

    result = max_2nd_derivative_detector(steps, vals, window_size=3)
    assert result['grokked'] is False
    assert result['cliff_step'] is None


def test_compute_median_iqr():
    vals = [10, 20, 30, 40, 50]
    med, q25, q75 = compute_median_iqr(vals)
    assert med == 30.0
    assert q25 == 20.0
    assert q75 == 40.0


def test_holm_mann_whitney():
    # Provide highly distinct groups to ensure significant results
    groups = {
        'pure': [1000, 1100, 1050, 950, 1000],
        'low': [2000, 2100, 2050, 1950, 2000],
        'high': [5000, 5100, 5050, 4950, 5000]
    }

    results = holm_mann_whitney(groups, 'pure')

    assert 'low' in results
    assert 'high' in results

    # High should have a very small p-value
    assert results['high']['p_adj'] < 0.05
    assert results['low']['p_adj'] < 0.05

    # p_adj should be >= p_raw
    assert results['high']['p_adj'] >= results['high']['p_raw']


def test_permutation_spearman_calibration():
    # Test that when variables are random, p-values are uniformly distributed
    # (Checking calibration by just making sure it doesn't always reject the null)
    rng = np.random.default_rng(123)
    p_vals = []

    # Run 20 small permutation tests
    for _ in range(20):
        x = rng.random(10)
        y = rng.random(10)
        _, p = permutation_spearman(x, y, num_permutations=200, seed=rng.integers(1000))
        p_vals.append(p)

    # Roughly uniform means about 5% should be < 0.05. Here we just assert it's not ALWAYS < 0.05
    reject_count = sum(1 for p in p_vals if p < 0.05)
    assert reject_count < 10 # Should be very rare


def test_permutation_spearman_strong_correlation():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # perfect correlation
    corr, p = permutation_spearman(x, y, num_permutations=1000, seed=42)
    assert corr > 0.99
    assert p < 0.01


def test_aggregate_and_plot(tmp_path):
    # Setup mock multi-seed directory structure
    results_dir = tmp_path / "multi_seed"

    # Seed 42, pure condition
    pure_dir = results_dir / "42" / "pure"
    pure_dir.mkdir(parents=True)

    steps = list(range(0, 5000, 50))
    pure_vals = generate_s_curve(steps, 2000, 0.005, 1.0).tolist()

    history = [{"step": s, "test_acc": v} for s, v in zip(steps, pure_vals)]
    history[-1]["weight_norm"] = 2.5

    with open(pure_dir / "results.json", 'w') as f:
        json.dump({"history": history, "final_test_acc": 1.0, "final_weight_norm": 2.5}, f)

    # Seed 42, high_collapse condition
    high_dir = results_dir / "42" / "high_collapse"
    high_dir.mkdir(parents=True)

    high_vals = generate_s_curve(steps, 2000, 0.005, 0.3).tolist()

    high_history = [{"step": s, "test_acc": v} for s, v in zip(steps, high_vals)]
    high_history[-1]["weight_norm"] = 0.5

    with open(high_dir / "results.json", 'w') as f:
        json.dump({"history": high_history, "final_test_acc": 0.3, "final_weight_norm": 0.5}, f)

    output_csv = tmp_path / "agg.csv"

    stats = aggregate_and_plot(
        results_dir=str(results_dir),
        output_csv=str(output_csv),
        window_size=3,
        control_condition="pure"
    )

    assert "pure" in stats['condition_stats']
    assert "high_collapse" in stats['condition_stats']

    assert stats['condition_stats']['pure']['grokked_runs'] == 1
    assert stats['condition_stats']['high_collapse']['grokked_runs'] == 0

    assert 'weight_norm_correlation' in stats

    assert output_csv.exists()
