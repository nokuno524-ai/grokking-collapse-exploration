import zlib
import numpy as np
import torch
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from collections import Counter
from typing import List, Union, Dict, Tuple

def _to_numpy_list(data: Union[torch.Tensor, np.ndarray, List[List[int]]]) -> List[List[int]]:
    """Convert input data to a list of lists of integers."""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(data, np.ndarray):
        data = data.tolist()
    return data

def ngram_diversity(data: Union[torch.Tensor, np.ndarray, List[List[int]]], max_n: int = 5) -> Dict[int, float]:
    """
    Compute n-gram diversity for n=1 to max_n.
    Returns the ratio of unique n-grams to total possible n-grams.
    If sequences are shorter than n, that n-gram diversity is 0.
    """
    data_list = _to_numpy_list(data)
    results = {}

    for n in range(1, max_n + 1):
        ngrams = []
        for seq in data_list:
            if len(seq) >= n:
                for i in range(len(seq) - n + 1):
                    ngrams.append(tuple(seq[i:i+n]))

        if not ngrams:
            results[n] = 0.0
        else:
            unique_ngrams = len(set(ngrams))
            results[n] = unique_ngrams / len(ngrams)

    return results

def token_distribution_shift(original_data: Union[torch.Tensor, np.ndarray, List[List[int]]],
                             synthetic_data: Union[torch.Tensor, np.ndarray, List[List[int]]],
                             vocab_size: int = None) -> float:
    """
    Compute KL divergence between token distributions of original and synthetic data.
    """
    orig_list = _to_numpy_list(original_data)
    synth_list = _to_numpy_list(synthetic_data)

    orig_tokens = [token for seq in orig_list for token in seq]
    synth_tokens = [token for seq in synth_list for token in seq]

    if not orig_tokens or not synth_tokens:
        return float('inf')

    if vocab_size is None:
        vocab_size = max(max(orig_tokens, default=0), max(synth_tokens, default=0)) + 1

    orig_counts = np.bincount(orig_tokens, minlength=vocab_size)
    synth_counts = np.bincount(synth_tokens, minlength=vocab_size)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    orig_probs = (orig_counts + epsilon) / (len(orig_tokens) + vocab_size * epsilon)
    synth_probs = (synth_counts + epsilon) / (len(synth_tokens) + vocab_size * epsilon)

    return float(entropy(synth_probs, orig_probs))

def sequence_length_comparison(original_data: Union[torch.Tensor, np.ndarray, List[List[int]]],
                               synthetic_data: Union[torch.Tensor, np.ndarray, List[List[int]]]) -> Dict[str, float]:
    """
    Compare sequence length distributions using KS test and Wasserstein distance.
    """
    orig_list = _to_numpy_list(original_data)
    synth_list = _to_numpy_list(synthetic_data)

    orig_lengths = [len(seq) for seq in orig_list]
    synth_lengths = [len(seq) for seq in synth_list]

    if not orig_lengths or not synth_lengths:
        return {"ks_statistic": 0.0, "ks_pvalue": 1.0, "wasserstein_distance": 0.0}

    ks_stat, ks_pval = ks_2samp(orig_lengths, synth_lengths)
    wd = wasserstein_distance(orig_lengths, synth_lengths)

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "wasserstein_distance": float(wd)
    }

def memorization_detection(train_data: Union[torch.Tensor, np.ndarray, List[List[int]]],
                           synthetic_data: Union[torch.Tensor, np.ndarray, List[List[int]]]) -> float:
    """
    Calculate fraction of synthetic sequences that exactly match any training sequence.
    """
    train_list = _to_numpy_list(train_data)
    synth_list = _to_numpy_list(synthetic_data)

    train_set = set(tuple(seq) for seq in train_list)

    if not synth_list:
        return 0.0

    matches = sum(1 for seq in synth_list if tuple(seq) in train_set)
    return float(matches / len(synth_list))

def _compress_size(data: bytes) -> int:
    return len(zlib.compress(data))

def ncd(seq1: List[int], seq2: List[int]) -> float:
    """Normalized Compression Distance between two sequences."""
    b1 = bytes(seq1)
    b2 = bytes(seq2)
    b12 = bytes(seq1 + seq2)

    c1 = _compress_size(b1)
    c2 = _compress_size(b2)
    c12 = _compress_size(b12)

    return (c12 - min(c1, c2)) / max(c1, c2)

def diversity_metrics(data: Union[torch.Tensor, np.ndarray, List[List[int]]]) -> Dict[str, float]:
    """
    Compute unique sequence fraction, type-token ratio, and mean NCD.
    """
    data_list = _to_numpy_list(data)

    if not data_list:
        return {"unique_fraction": 0.0, "type_token_ratio": 0.0, "mean_ncd": 0.0}

    # Unique sequence fraction
    unique_seqs = len(set(tuple(seq) for seq in data_list))
    unique_fraction = unique_seqs / len(data_list)

    # Type-token ratio
    all_tokens = [token for seq in data_list for token in seq]
    if all_tokens:
        unique_tokens = len(set(all_tokens))
        ttr = unique_tokens / len(all_tokens)
    else:
        ttr = 0.0

    # NCD (sample up to 100 random pairs to avoid O(N^2))
    ncd_vals = []
    n_samples = min(len(data_list), 100)
    if n_samples > 1:
        indices = np.random.choice(len(data_list), n_samples, replace=False)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                ncd_vals.append(ncd(data_list[indices[i]], data_list[indices[j]]))

    mean_ncd = float(np.mean(ncd_vals)) if ncd_vals else 0.0

    return {
        "unique_fraction": float(unique_fraction),
        "type_token_ratio": float(ttr),
        "mean_ncd": float(mean_ncd)
    }
