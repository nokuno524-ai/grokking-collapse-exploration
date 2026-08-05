import torch
import numpy as np
from src.data import generate_modular_arithmetic, DatasetConfig, apply_collapse, apply_label_noise

def test_generate_modular_arithmetic_pure():
    config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0, noise_fraction=0.0)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_samples = 59 * 59
    expected_train = int(round(total_samples * 0.3))
    expected_test = total_samples - expected_train

    assert train_in.shape[0] == expected_train
    assert train_tgt.shape[0] == expected_train
    assert test_in.shape[0] == expected_test
    assert test_tgt.shape[0] == expected_test

    # Check that outputs match modular addition
    for i in range(10):
        a, b = train_in[i].tolist()
        assert train_tgt[i].item() == (a + b) % 59

def test_apply_collapse():
    rng = np.random.RandomState(42)
    prime = 59
    pairs = [(a, b) for a in range(10) for b in range(10)]
    targets = [(a + b) % prime for a, b in pairs]
    collapse_level = 0.5
    collapse_severity = 0.9

    new_pairs, new_targets = apply_collapse(pairs, targets, prime, collapse_level, collapse_severity, rng)

    assert len(new_pairs) == len(pairs)
    assert len(new_targets) == len(targets)

    # Count how many changed
    changes = sum(1 for t1, t2 in zip(targets, new_targets) if t1 != t2)
    assert changes > 0

def test_apply_label_noise():
    rng = np.random.RandomState(42)
    prime = 59
    pairs = [(a, b) for a in range(10) for b in range(10)]
    targets = [(a + b) % prime for a, b in pairs]
    noise_fraction = 0.3

    new_pairs, new_targets = apply_label_noise(pairs, targets, prime, noise_fraction, rng)

    assert len(new_pairs) == len(pairs)
    assert len(new_targets) == len(targets)

    # Count how many changed
    changes = sum(1 for t1, t2 in zip(targets, new_targets) if t1 != t2)
    expected_changes = int(round(len(targets) * noise_fraction))
    assert changes == expected_changes
