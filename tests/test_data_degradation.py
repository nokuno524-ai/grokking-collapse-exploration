import pytest
import numpy as np
from src.analysis.data_degradation import (
    compute_entropy, compute_distinct_n, compute_compression_ratio,
    compute_jensen_shannon_divergence, compute_duplicate_fraction,
    compute_data_degradation_metrics
)

def test_entropy():
    # Uniform distribution (2 tokens, equal freq) -> entropy of 1.0
    corpus = [0, 1, 0, 1]
    assert np.isclose(compute_entropy(corpus), 1.0)

    # Single token -> entropy of 0.0
    corpus = [0, 0, 0, 0]
    assert np.isclose(compute_entropy(corpus), 0.0)

def test_distinct_n():
    corpus = [0, 1, 2, 3] # distinct-1 = 4/4 = 1.0, distinct-2 = 3/3 = 1.0
    assert compute_distinct_n(corpus, n=1) == 1.0
    assert compute_distinct_n(corpus, n=2) == 1.0

    corpus = [0, 0, 0, 0] # distinct-1 = 1/4 = 0.25, distinct-2 = 1/3
    assert compute_distinct_n(corpus, n=1) == 0.25
    assert np.isclose(compute_distinct_n(corpus, n=2), 1/3)

def test_duplicate_fraction():
    corpus = [0, 1, 2, 3]
    assert compute_duplicate_fraction(corpus) == 0.0

    corpus = [0, 0, 0, 0]
    assert compute_duplicate_fraction(corpus) == 0.75 # 3 duplicates out of 4

def test_js_divergence():
    p = [0.5, 0.5]
    q = [0.5, 0.5]
    assert np.isclose(compute_jensen_shannon_divergence(p, q), 0.0)

    p = [1.0, 0.0]
    q = [0.0, 1.0]
    assert compute_jensen_shannon_divergence(p, q) > 0.0

def test_compute_all_metrics():
    current_corpus = [0, 0, 1, 1]
    base_corpus = [0, 1, 2, 3]
    vocab_size = 4

    metrics = compute_data_degradation_metrics(current_corpus, base_corpus, vocab_size)
    assert 'entropy' in metrics
    assert 'distinct_1' in metrics
    assert 'distinct_2' in metrics
    assert 'compression_ratio' in metrics
    assert 'duplicate_fraction' in metrics
    assert 'js_divergence_from_base' in metrics

def test_cli_fixture(tmp_path):
    from src.cli_data_degradation import run_sweep_and_compute_metrics
    import os

    output_dir = str(tmp_path)
    df = run_sweep_and_compute_metrics(output_dir, max_depth=1, seed=42)

    assert len(df) > 0
    assert os.path.exists(os.path.join(output_dir, 'data_degradation_metrics.csv'))
    assert os.path.exists(os.path.join(output_dir, 'entropy_vs_generation.png'))
