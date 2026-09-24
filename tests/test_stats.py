import numpy as np
import pytest
import math
from src.rigorous_stats.analyzer import (
    compute_bootstrap_ci,
    compute_survival_analysis,
    permutation_test
)

def test_compute_bootstrap_ci():
    # Known dataset
    # We create an array with a known median
    data = [1, 2, 3, 4, 5, 6, 7]
    median, lower, upper = compute_bootstrap_ci(data, num_samples=100)

    assert median == 4.0
    # Bootstrap CIs should bound the median
    assert lower <= median
    assert upper >= median

    # Test empty list
    med_empty, low_empty, up_empty = compute_bootstrap_ci([])
    assert math.isnan(med_empty)

def test_permutation_test():
    # Two very distinct distributions
    group_a = [1, 2, 3, 1, 2, 3]
    group_b = [100, 101, 102, 100, 101, 102]

    p_val = permutation_test(group_a, group_b, n_permutations=100)

    # Since they are completely distinct, p-value should be very low, likely 0
    assert p_val < 0.05

    # Empty groups
    assert math.isnan(permutation_test([], [1, 2]))

def test_survival_analysis():
    runs = [
        {"grokking_step": 1000},
        {"grokking_step": 1200},
        {"grokking_step": None}, # Censored
        {"grokking_step": 1500},
        {"grokking_step": None}  # Censored
    ]

    stats = compute_survival_analysis(runs, max_steps=50000)

    assert stats['grok_fraction'] == 3/5
    assert stats['times'] == [1000, 1200, 50000, 1500, 50000]
    assert stats['events'] == [1, 1, 0, 1, 0]

    # Median of [1000, 1200, 1500] is 1200
    assert stats['median'] == 1200.0
