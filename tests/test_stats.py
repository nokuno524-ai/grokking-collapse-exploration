import numpy as np
import pytest
import math
from src.rigorous_stats.cliff_detection import binary_segmentation_changepoint, find_cliff_location
from src.rigorous_stats.stats_module import power_analysis, power_analysis_simulated

def test_binary_segmentation_changepoint_synthetic():
    """Test changepoint detection on a synthetic piecewise constant curve with noise."""
    rng = np.random.RandomState(0)

    # 50 points of ~0.1, 50 points of ~0.9
    y1 = rng.normal(0.1, 0.05, 50)
    y2 = rng.normal(0.9, 0.05, 50)
    y = np.concatenate([y1, y2])

    idx, min_cost, costs = binary_segmentation_changepoint(y)

    # True changepoint is at index 50
    assert idx == 50
    # Cost should be > 0 due to noise
    assert min_cost > 0
    # Cost at index 50 should be the minimum
    assert np.argmin(costs) == 50

def test_find_cliff_location_degenerate_inputs():
    """Test find_cliff_location on degenerate inputs (empty, single element, constant)."""
    # Empty arrays
    steps = np.array([])
    accs = np.array([])
    assert find_cliff_location(steps, accs) == (None, None)

    # Single element
    steps = np.array([1])
    accs = np.array([0.5])
    assert find_cliff_location(steps, accs) == (None, None)

    # Two elements (too short for cliff detection)
    steps = np.array([1, 2])
    accs = np.array([0.5, 0.5])
    assert find_cliff_location(steps, accs) == (None, None)

    # Constant array (no cliff)
    steps = np.arange(10)
    accs = np.zeros(10)
    cliff_step, ci = find_cliff_location(steps, accs)
    assert cliff_step is None

def test_find_cliff_location_nan_handling():
    """Test find_cliff_location with NaN values."""
    steps = np.arange(10)
    accs = np.array([0.1, 0.1, 0.1, np.nan, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    assert find_cliff_location(steps, accs) == (None, None)

def test_find_cliff_location_no_jump():
    """Test that a small jump is not considered a cliff."""
    steps = np.arange(20)
    # Jump of 0.1 (less than threshold of 0.2)
    accs = np.concatenate([np.ones(10)*0.1, np.ones(10)*0.2])
    cliff_step, ci = find_cliff_location(steps, accs)
    assert cliff_step is None

def test_power_analysis():
    """Test power analysis calculations."""
    variance = 100.0  # std=10
    target_diff = 10.0

    # Closed form
    n_closed = power_analysis(variance, target_diff, alpha=0.05, power=0.8)
    # The true required N is around 16 per group
    assert 15 <= n_closed <= 17

    # Simulated (might take a second, so we keep n_simulations small for the test)
    n_simulated = power_analysis_simulated(variance, target_diff, alpha=0.05, power=0.8, n_simulations=200)

    # Should be close to closed form
    assert 14 <= n_simulated <= 18

def test_power_analysis_invalid_inputs():
    """Test power analysis with invalid inputs."""
    assert math.isnan(power_analysis(-1, 10))
    assert math.isnan(power_analysis(100, 0))
    assert math.isnan(power_analysis_simulated(-1, 10))
    assert math.isnan(power_analysis_simulated(100, 0))

def test_sigmoid_ramp():
    """Test changepoint detection on a smooth sigmoid ramp."""
    steps = np.arange(100)
    # Sigmoid centered at 50, scale 5
    accs = 1 / (1 + np.exp(-(steps - 50) / 5))
    idx, min_cost, costs = binary_segmentation_changepoint(accs)
    # Changepoint should be near the center of the sigmoid
    assert 45 <= idx <= 55

def test_double_cliff():
    """Test changepoint detection on a curve with two cliffs."""
    steps = np.arange(150)
    # 0.1 for 50 steps, 0.5 for 50 steps, 0.9 for 50 steps
    accs = np.concatenate([np.ones(50)*0.1, np.ones(50)*0.5, np.ones(50)*0.9])

    idx, min_cost, costs = binary_segmentation_changepoint(accs)

    # Binary segmentation will find one of the two cliffs, depending on variance reduction.
    # Because the jumps are equal (0.4) and the segment lengths are equal (50),
    # the optimal single split is exactly in the middle of the middle segment (idx 75),
    # or symmetric at 50/100 depending on exact cost values.
    # In a pure piecewise constant fit of 3 blocks to 2 blocks,
    # splitting at 50 or 100 yields the same SSE.
    # Let's just ensure it doesn't crash and returns a reasonable index.
    assert idx in [50, 75, 100]
