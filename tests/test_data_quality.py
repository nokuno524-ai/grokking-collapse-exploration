import pytest
import torch
import numpy as np
from src.data import DatasetConfig, generate_modular_arithmetic, apply_collapse
from collections import Counter
import scipy.stats as stats

@pytest.mark.parametrize("prime", [17, 59, 97])
def test_modular_arithmetic_correctness(prime):
    """Verify a+b mod p is correct for various p."""
    config = DatasetConfig(prime=prime, train_fraction=1.0, collapse_level=0.0)
    train_in, train_tgt, _, _ = generate_modular_arithmetic(config)

    # Check all
    for i in range(len(train_in)):
        a, b = train_in[i].tolist()
        expected = (a + b) % prime
        assert train_tgt[i].item() == expected

def test_data_leakage():
    """Verify train/test split has no leakage."""
    config = DatasetConfig(prime=59, train_fraction=0.3, collapse_level=0.0)
    train_in, _, test_in, _ = generate_modular_arithmetic(config)

    train_set = set(tuple(x.tolist()) for x in train_in)
    test_set = set(tuple(x.tolist()) for x in test_in)

    # Intersection should be empty
    assert len(train_set.intersection(test_set)) == 0
    # Union should be all possible pairs
    assert len(train_set.union(test_set)) == 59 * 59

def test_collapse_distribution():
    """Verify collapse distribution changes target frequencies and calculate KL divergence."""
    prime = 59
    pairs = [(a, b) for a in range(prime) for b in range(prime)]
    targets = [(a + b) % prime for a, b in pairs]

    rng = np.random.RandomState(42)

    # Apply severe collapse
    new_pairs, new_targets = apply_collapse(pairs, targets, prime, collapse_level=1.0, collapse_severity=0.9, rng=rng)

    orig_counts = Counter(targets)
    orig_freqs = np.array([orig_counts[i] for i in range(prime)], dtype=float)
    orig_probs = orig_freqs / orig_freqs.sum()

    new_counts = Counter(new_targets)
    new_freqs = np.array([new_counts.get(i, 1e-10) for i in range(prime)], dtype=float)
    new_probs = new_freqs / new_freqs.sum()

    # Calculate KL divergence (should be > 0 due to collapse)
    kl_div = stats.entropy(new_probs, orig_probs)

    assert kl_div > 0.005

def test_label_uniformity_pure():
    """Statistical test on data quality (uniformity of labels)."""
    config = DatasetConfig(prime=59, train_fraction=1.0, collapse_level=0.0)
    _, train_tgt, _, _ = generate_modular_arithmetic(config)

    counts = Counter(train_tgt.tolist())
    freqs = [counts[i] for i in range(59)]

    # Expected is uniform
    expected = [len(train_tgt) / 59] * 59

    # Chi-square test
    chi2, p_value = stats.chisquare(freqs, f_exp=expected)
    # Should not reject null hypothesis (p > 0.05)
    assert p_value > 0.05
