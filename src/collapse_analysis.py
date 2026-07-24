import numpy as np
import scipy.stats
from collections import Counter
from typing import List, Dict, Any, Tuple

def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the KL divergence KL(P || Q) between two discrete probability distributions.
    Adds a small epsilon to avoid division by zero or log(0).
    """
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))

def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the Jensen-Shannon divergence between two discrete probability distributions.
    """
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)

def get_distribution(tokens: List[int], vocab_size: int) -> np.ndarray:
    """
    Returns the probability distribution of tokens.
    """
    counts = np.bincount(tokens, minlength=vocab_size)
    total = np.sum(counts)
    if total == 0:
        return np.ones(vocab_size) / vocab_size
    return counts / total

def analyze_mode_collapse(outputs: List[List[int]], vocab_size: int) -> Dict[str, float]:
    """
    Tracks mode collapse indicators: vocabulary size, entropy of output distribution, n-gram diversity.
    Assumes outputs is a list of token sequences.
    """
    if not outputs:
        return {"vocab_used": 0.0, "entropy": 0.0, "unigram_diversity": 0.0, "bigram_diversity": 0.0}

    flat_tokens = [token for seq in outputs for token in seq]

    if not flat_tokens:
        return {"vocab_used": 0.0, "entropy": 0.0, "unigram_diversity": 0.0, "bigram_diversity": 0.0}

    # Vocabulary used
    unique_tokens = set(flat_tokens)
    vocab_used = len(unique_tokens) / vocab_size

    # Entropy of output distribution
    dist = get_distribution(flat_tokens, vocab_size)
    entropy = scipy.stats.entropy(dist)

    # Unigram diversity (unique unigrams / total unigrams)
    unigram_diversity = len(unique_tokens) / len(flat_tokens) if flat_tokens else 0.0

    # Bigram diversity
    bigrams = list(zip(flat_tokens[:-1], flat_tokens[1:]))
    unique_bigrams = set(bigrams)
    bigram_diversity = len(unique_bigrams) / len(bigrams) if bigrams else 0.0

    return {
        "vocab_used_fraction": vocab_used,
        "entropy": float(entropy),
        "unigram_diversity": unigram_diversity,
        "bigram_diversity": bigram_diversity
    }

def compute_distributional_shift(base_data: List[int], collapse_data: List[int], vocab_size: int) -> Dict[str, float]:
    """
    Computes distributional shift metrics (KL, JS) between a base distribution and a collapsed distribution.
    """
    p = get_distribution(base_data, vocab_size)
    q = get_distribution(collapse_data, vocab_size)

    kl = compute_kl_divergence(p, q)
    js = compute_js_divergence(p, q)

    return {
        "kl_divergence": kl,
        "js_divergence": js
    }

def correlate_collapse_with_head_specialization(
    collapse_severities: List[float],
    head_specializations: List[float]
) -> float:
    """
    Computes Pearson correlation between collapse severity and attention head specialization.
    """
    if len(collapse_severities) < 2 or len(head_specializations) < 2:
        return 0.0

    r, _ = scipy.stats.pearsonr(collapse_severities, head_specializations)

    # Handle NaN if constant
    if np.isnan(r):
        return 0.0

    return float(r)
