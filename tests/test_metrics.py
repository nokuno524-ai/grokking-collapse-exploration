import pytest
import torch
import numpy as np
from metrics.data_quality import (
    ngram_diversity,
    token_distribution_shift,
    sequence_length_comparison,
    memorization_detection,
    diversity_metrics,
    ncd
)

@pytest.fixture
def identical_data():
    """Returns two identical datasets."""
    # 10 sequences of length 3 (a, b, target)
    data = [
        [1, 2, 3], [4, 5, 9], [10, 11, 21], [7, 8, 15], [3, 4, 7],
        [1, 2, 3], [2, 2, 4], [5, 5, 10], [1, 1, 2], [9, 9, 18]
    ]
    return data, list(data)

@pytest.fixture
def structured_data():
    """Returns structured data resembling modular arithmetic."""
    return [
        [1, 2, 3], [4, 5, 9], [10, 11, 21], [7, 8, 15], [3, 4, 7]
    ]

@pytest.fixture
def random_data():
    """Returns random noisy data of same shape."""
    np.random.seed(42)
    return np.random.randint(0, 59, size=(5, 3)).tolist()

def test_identical_data_perfect_scores(identical_data):
    orig, synth = identical_data

    # KL divergence should be 0.0 (or very close to it)
    kl = token_distribution_shift(orig, synth)
    assert kl < 1e-5

    # Memorization should be 1.0 (100% exact match)
    mem = memorization_detection(orig, synth)
    assert mem == 1.0

    # Sequence length comparison should yield KS stat=0, p-val=1.0, WD=0
    seq_metrics = sequence_length_comparison(orig, synth)
    assert seq_metrics["ks_statistic"] == 0.0
    assert seq_metrics["ks_pvalue"] == 1.0
    assert seq_metrics["wasserstein_distance"] == 0.0

def test_random_data_poor_scores(structured_data, random_data):
    # KL divergence should be > 0
    kl = token_distribution_shift(structured_data, random_data)
    assert kl > 0.1

    # Memorization should be 0.0
    mem = memorization_detection(structured_data, random_data)
    assert mem == 0.0

def test_ngram_diversity(structured_data):
    # Check n-gram lengths up to 3
    div = ngram_diversity(structured_data, max_n=3)
    assert len(div) == 3
    assert 1 in div and 2 in div and 3 in div

    # Since there are 5 sequences of length 3:
    # 5 unique 3-grams
    assert div[3] == 1.0

def test_diversity_metrics(identical_data):
    data, _ = identical_data
    div = diversity_metrics(data)

    # Out of 10 sequences, [1, 2, 3] is repeated once
    # So 9 unique sequences out of 10 -> 0.9
    assert div["unique_fraction"] == 0.9

    assert "type_token_ratio" in div
    assert "mean_ncd" in div
    assert div["type_token_ratio"] <= 1.0
    assert div["mean_ncd"] >= 0.0

def test_ncd():
    # Identical sequences should compress similarly and yield NCD ~ 0.0
    # In practice with zlib, it's not exactly 0, but should be low
    seq1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seq2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert ncd(seq1, seq2) < 0.2

    # Completely different sequences should have higher NCD
    seq3 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    assert ncd(seq1, seq3) > ncd(seq1, seq2)
