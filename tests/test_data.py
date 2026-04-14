import torch
import numpy as np
from src.data import generate_modular_arithmetic, DatasetConfig, apply_collapse

def test_generate_modular_arithmetic_pure():
    config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0, seed=42)
    train_in, train_tgt, test_in, test_tgt = generate_modular_arithmetic(config)

    total_pairs = 59 * 59
    n_train = int(total_pairs * 0.3)
    n_test = total_pairs - n_train

    assert train_in.shape == (n_train, 2)
    assert train_tgt.shape == (n_train,)
    assert test_in.shape == (n_test, 2)
    assert test_tgt.shape == (n_test,)

    assert train_in.dtype == torch.long
    assert train_tgt.dtype == torch.long

    assert torch.all((train_in[:, 0] + train_in[:, 1]) % 59 == train_tgt)

def test_apply_collapse():
    prime = 59
    rng = np.random.RandomState(42)

    pairs = [(a, b) for a in range(prime) for b in range(prime)]
    targets = [(a + b) % prime for a, b in pairs]

    new_pairs, new_targets = apply_collapse(
        pairs, targets, prime, collapse_level=0.5, collapse_severity=0.5, rng=rng
    )

    assert len(new_pairs) == len(pairs)
    assert len(new_targets) == len(targets)

    diff = sum(1 for t1, t2 in zip(targets, new_targets) if t1 != t2)
    assert diff > 0
