import pytest
from src.data.metrics import compute_entropy, compute_distinct_n, compute_repetition_rate, compute_zipf_coefficient, compute_jensen_shannon_divergence

def test_compute_entropy():
    # Uniform distribution
    counts = {1: 10, 2: 10, 3: 10, 4: 10}
    assert compute_entropy(counts) > 1.3

    # Degenerate distribution
    counts = {1: 40}
    assert compute_entropy(counts) == 0.0

def test_compute_distinct_n():
    seq = [1, 2, 3, 4, 5]
    assert compute_distinct_n(seq, 1) == 1.0

    seq = [1, 1, 1, 1, 1]
    assert compute_distinct_n(seq, 1) == 0.2
    assert compute_distinct_n(seq, 2) == 0.25

def test_compute_repetition_rate():
    seq = [1, 2, 3, 4, 5]
    assert compute_repetition_rate(seq) == 0.0

    seq = [1, 1, 2, 2, 3]
    assert compute_repetition_rate(seq) == 0.5

    seq = [1, 1, 1, 1, 1]
    assert compute_repetition_rate(seq) == 1.0

def test_compute_jensen_shannon_divergence():
    p = {1: 10, 2: 10}
    q = {1: 10, 2: 10}
    assert compute_jensen_shannon_divergence(p, q, 3) == 0.0

    p = {1: 20}
    q = {2: 20}
    assert compute_jensen_shannon_divergence(p, q, 3) > 0.6
