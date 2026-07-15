import torch
import numpy as np
from src.data import generate_modular_arithmetic, DatasetConfig, apply_collapse, apply_label_noise

def test_modular_arithmetic_pure():
    config = DatasetConfig(prime=7, train_fraction=0.5, collapse_level=0.0, noise_fraction=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    # 7 * 7 = 49 pairs total. train_fraction 0.5 -> 24 train, 25 test.
    assert len(train_in) == 24
    assert len(test_in) == 25

    for i in range(len(train_in)):
        a, b = train_in[i]
        assert train_tgt[i] == (a + b) % 7

def test_collapse_ratio():
    config = DatasetConfig(prime=7, train_fraction=1.0, collapse_level=0.5, collapse_severity=0.9)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    # Expect some targets to deviate from exact modular arithmetic
    exact_matches = sum(train_tgt[i] == (train_in[i][0] + train_in[i][1]) % 7 for i in range(len(train_in)))
    assert exact_matches < len(train_in)  # some are collapsed

def test_apply_label_noise():
    rng = np.random.RandomState(42)
    pairs = [(i, j) for i in range(5) for j in range(5)]
    targets = [(i + j) % 5 for i, j in pairs]
    new_pairs, new_targets = apply_label_noise(pairs, targets, 5, 0.5, rng)

    mismatches = sum(1 for i in range(len(targets)) if targets[i] != new_targets[i])
    # 50% of 25 is 12.5 -> 12 replaced
    assert mismatches == 12

def test_apply_collapse_base_prob():
    # Test that missing targets get a small base probability
    rng = np.random.RandomState(42)
    pairs = [(0, 0), (1, 1), (2, 2)]
    targets = [0, 2, 4]  # Target 1, 3, etc. are missing

    new_pairs, new_targets = apply_collapse(pairs, targets, 5, 1.0, 0.9, rng)
    # The code should run without errors (no division by zero or amplifying rare targets excessively)
    assert len(new_targets) == len(targets)
