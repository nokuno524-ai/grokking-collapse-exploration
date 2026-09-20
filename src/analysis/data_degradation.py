import numpy as np
import collections
import zlib
import pandas as pd
from typing import List, Dict, Any, Tuple
from scipy.stats import entropy
import os
import glob
import json

def compute_entropy(corpus: List[int]) -> float:
    """Computes Shannon entropy of the vocabulary/fingerprint in a corpus."""
    if not corpus:
        return 0.0
    counts = collections.Counter(corpus)
    probs = [c / len(corpus) for c in counts.values()]
    return float(entropy(probs, base=2))

def compute_distinct_n(corpus: List[int], n: int = 2) -> float:
    """Computes distinct-n metric (ratio of unique n-grams to total n-grams)."""
    if len(corpus) < n:
        return 0.0
    ngrams = [tuple(corpus[i:i+n]) for i in range(len(corpus) - n + 1)]
    return len(set(ngrams)) / len(ngrams)

def compute_compression_ratio(corpus: List[int]) -> float:
    """Computes normalized gzip length to evaluate data compression/redundancy."""
    if not corpus:
        return 1.0
    # Convert corpus to string for compression
    corpus_str = " ".join(map(str, corpus)).encode('utf-8')
    compressed = zlib.compress(corpus_str)
    return len(compressed) / len(corpus_str)

def compute_jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    """Computes the Jensen-Shannon divergence between two distributions."""
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)

def extract_distribution(corpus: List[int], vocab_size: int) -> List[float]:
    """Extracts a probability distribution over the vocabulary."""
    counts = collections.Counter(corpus)
    dist = [counts.get(i, 0) / len(corpus) for i in range(vocab_size)]
    return dist

def compute_duplicate_fraction(corpus: List[int]) -> float:
    """Computes the fraction of duplicate tokens/items in the corpus."""
    if not corpus:
        return 0.0
    return 1.0 - (len(set(corpus)) / len(corpus))


def compute_length_distribution_drift(current_lengths: List[int], base_lengths: List[int]) -> float:
    """Computes Jensen-Shannon divergence between length distributions."""
    if not current_lengths or not base_lengths:
        return 0.0

    max_len = max(max(current_lengths), max(base_lengths))

    current_dist = extract_distribution(current_lengths, max_len + 1)
    base_dist = extract_distribution(base_lengths, max_len + 1)

    return compute_jensen_shannon_divergence(current_dist, base_dist)

def compute_data_degradation_metrics(
    current_corpus: List[int],
    base_corpus: List[int],
    vocab_size: int
) -> Dict[str, float]:
    """Computes a suite of data degradation metrics for a given corpus."""

    metrics = {}

    # Vocabulary/fingerprint entropy
    metrics['entropy'] = compute_entropy(current_corpus)

    # n-gram diversity (distinct-1, distinct-2)
    metrics['distinct_1'] = compute_distinct_n(current_corpus, n=1)
    metrics['distinct_2'] = compute_distinct_n(current_corpus, n=2)

    # Compression ratio
    metrics['compression_ratio'] = compute_compression_ratio(current_corpus)

    # Duplicate fraction
    metrics['duplicate_fraction'] = compute_duplicate_fraction(current_corpus)

    # Distributional distance (Jensen-Shannon)
    current_dist = extract_distribution(current_corpus, vocab_size)
    base_dist = extract_distribution(base_corpus, vocab_size)
    metrics['js_divergence_from_base'] = compute_jensen_shannon_divergence(current_dist, base_dist)

    return metrics
