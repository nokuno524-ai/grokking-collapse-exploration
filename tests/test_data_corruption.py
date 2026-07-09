import pytest
import numpy as np
from src.data import apply_collapse, apply_label_noise

def test_apply_collapse():
    prime = 59
    rng = np.random.RandomState(42)
    # Generate some structured data
    pairs = [(a, b) for a in range(10) for b in range(10)]
    targets = [(a + b) % prime for a, b in pairs]

    collapse_level = 0.5
    collapse_severity = 0.8

    new_pairs, new_targets = apply_collapse(
        pairs, targets, prime, collapse_level, collapse_severity, rng
    )

    # 50% should be corrupted
    diff_count = sum(1 for t1, t2 in zip(targets, new_targets) if t1 != t2)
    assert diff_count > 0, "No targets were corrupted by collapse."
    assert len(new_pairs) == len(pairs)
    assert len(new_targets) == len(targets)

def test_apply_label_noise():
    prime = 59
    rng = np.random.RandomState(42)
    pairs = [(a, b) for a in range(10) for b in range(10)]
    targets = [(a + b) % prime for a, b in pairs]

    noise_fraction = 0.3

    new_pairs, new_targets = apply_label_noise(
        pairs, targets, prime, noise_fraction, rng
    )

    diff_count = sum(1 for t1, t2 in zip(targets, new_targets) if t1 != t2)

    # Check that exactly (or approximately) 30% are different
    n_replace = int(len(targets) * noise_fraction)
    assert diff_count == n_replace, f"Expected {n_replace} replacements, got {diff_count}"
    assert len(new_pairs) == len(pairs)
    assert len(new_targets) == len(targets)
