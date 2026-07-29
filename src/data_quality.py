import numpy as np
import torch
import collections
from collections import Counter
from typing import List, Union, Dict, Tuple
import math

def ngram_diversity(tokens: List[int], max_n: int = 4) -> Dict[int, float]:
    """
    Computes unique n-grams / total n-grams for n=1, 2, 3, 4.
    """
    diversity = {}
    for n in range(1, max_n + 1):
        if len(tokens) < n:
            diversity[n] = 0.0
            continue

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))

        unique_ngrams = set(ngrams)
        diversity[n] = len(unique_ngrams) / len(ngrams) if ngrams else 0.0

    return diversity

def token_frequency_analysis(tokens: List[int], reference_distribution: Dict[int, float] = None) -> Dict[str, float]:
    """
    Computes Zipf coefficient and KL divergence from original/reference distribution.
    """
    if not tokens:
        return {"zipf_coefficient": 0.0, "kl_divergence": 0.0}

    counts = Counter(tokens)
    total_tokens = len(tokens)

    # Zipf coefficient estimation
    # Rank frequencies
    sorted_counts = sorted(counts.values(), reverse=True)
    if len(sorted_counts) > 1:
        # Simple estimation of zipf parameter s using linear regression on log-log scale
        # log(f_k) = log(C) - s * log(k)
        ranks = np.arange(1, len(sorted_counts) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(sorted_counts)

        # Fit line
        slope, _ = np.polyfit(log_ranks, log_freqs, 1)
        zipf_coeff = -slope
    else:
        zipf_coeff = 0.0

    # KL Divergence
    kl_div = 0.0
    if reference_distribution is not None:
        for token, count in counts.items():
            p_x = count / total_tokens
            q_x = reference_distribution.get(token, 1e-10) # small smoothing
            if q_x > 0 and p_x > 0:
                kl_div += p_x * math.log(p_x / q_x)

    return {
        "zipf_coefficient": float(zipf_coeff),
        "kl_divergence": float(kl_div)
    }

def perplexity_estimation(tokens: List[int]) -> float:
    """
    Estimates perplexity using a simple bigram reference model on the data itself.
    (In a real scenario, this would use a held-out small reference LM).
    """
    if len(tokens) < 2:
        return 0.0

    bigram_counts = Counter()
    unigram_counts = Counter()

    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        bigram_counts[bigram] += 1
        unigram_counts[tokens[i]] += 1

    # Last token
    unigram_counts[tokens[-1]] += 1

    vocab_size = len(unigram_counts)
    log_prob_sum = 0.0

    # Calculate log probability of the sequence
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        # Add-1 smoothing
        prob = (bigram_counts[bigram] + 1) / (unigram_counts[tokens[i]] + vocab_size)
        log_prob_sum += math.log(prob)

    avg_log_prob = log_prob_sum / (len(tokens) - 1)

    try:
        perplexity = math.exp(-avg_log_prob)
    except OverflowError:
        perplexity = float('inf')

    return perplexity

def repetition_detection(tokens: List[int], sequence_length: int = 10) -> float:
    """
    Calculates fraction of repeated sequences of a specific length.
    """
    if len(tokens) < sequence_length:
        return 0.0

    sequences = []
    for i in range(len(tokens) - sequence_length + 1):
        sequences.append(tuple(tokens[i:i+sequence_length]))

    counts = Counter(sequences)
    repeated = sum(1 for count in counts.values() if count > 1)

    return repeated / len(counts) if counts else 0.0

def collapse_score(tokens: List[int], reference_distribution: Dict[int, float] = None) -> float:
    """
    Produces a composite metric combining all signals.
    Higher score indicates more collapse.
    """
    if not tokens:
        return 0.0

    div = ngram_diversity(tokens, max_n=4)
    # Average diversity (lower is more collapsed)
    avg_div = sum(div.values()) / len(div)

    freq_analysis = token_frequency_analysis(tokens, reference_distribution)
    # Higher Zipf means steeper distribution (more collapsed to few tokens)
    zipf = freq_analysis["zipf_coefficient"]
    # Higher KL divergence means further from reference (more collapsed)
    kl = freq_analysis["kl_divergence"]

    ppl = perplexity_estimation(tokens)
    # Lower perplexity implies more predictable/collapsed text
    # We invert it for the score (higher means more collapsed)
    inv_ppl = 1.0 / ppl if ppl > 0 else 0.0

    rep = repetition_detection(tokens)

    # Simple composite score (weights are arbitrary for demonstration)
    # We want:
    # - low avg_div -> high collapse
    # - high zipf -> high collapse
    # - high kl -> high collapse
    # - low ppl -> high collapse
    # - high rep -> high collapse

    score = (1.0 - avg_div) + 0.1 * zipf + 0.5 * kl + 10.0 * inv_ppl + 2.0 * rep

    return max(0.0, float(score))
