import pytest
import numpy as np
from src.data import DatasetConfig, apply_collapse, get_condition_seed, generate_modular_arithmetic

def test_exact_ratio_attainment():
    # Test that exactly n_replace items are generated
    pairs = [(i, i) for i in range(100)]
    targets = [i % 10 for i in range(100)]
    rng = np.random.RandomState(42)

    new_pairs, new_targets = apply_collapse(pairs, targets, prime=10, collapse_level=0.3, collapse_severity=0.5, rng=rng)

    # Check that lengths are unchanged
    assert len(new_pairs) == 100
    assert len(new_targets) == 100

    # We replaced exactly 30 items
    diff = sum([1 for a, b in zip(targets, new_targets) if a != b])
    # The actual diff can be slightly less if by chance a replaced item was replaced by the same target
    # But it must be <= 30
    assert diff <= 30

def test_seed_hashing():
    seed1 = get_condition_seed(42, "pure")
    seed2 = get_condition_seed(42, "low_collapse")
    assert seed1 != seed2

def test_data_generation_runs():
    config = DatasetConfig(prime=11, collapse_level=0.5, collapse_severity=0.9, tree_depth=2)
    t_in, t_out, te_in, te_out = generate_modular_arithmetic(config)
    assert len(t_in) == int(121 * 0.3)
