import numpy as np
from scipy.special import rel_entr
from typing import Dict, List, Tuple

def calculate_attention_entropy(attention_weights: np.ndarray) -> np.ndarray:
    """
    Computes Shannon entropy of attention distributions.

    Args:
        attention_weights: shape (batch_size, n_heads, seq_len, seq_len).
                           The last dimension is a probability distribution (sums to 1).

    Returns:
        entropy: shape (batch_size, n_heads).
                 Averaged over query positions (seq_len).
    """
    # Clip to avoid log(0)
    eps = 1e-10
    w = np.clip(attention_weights, eps, 1.0)

    # Entropy per query: -sum(p * log(p)) over keys
    entropy_per_query = -np.sum(w * np.log(w), axis=-1)  # (batch_size, n_heads, seq_len)

    # Average over queries
    return np.mean(entropy_per_query, axis=-1)  # (batch_size, n_heads)

def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes Jensen-Shannon Divergence between two probability distributions.

    Args:
        p: Array of shape (N,) summing to 1
        q: Array of shape (N,) summing to 1

    Returns:
        JS divergence (float)
    """
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)
    js = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
    return float(js)

def compute_head_similarity(attention_weights: np.ndarray) -> np.ndarray:
    """
    Computes similarity between all heads in a layer based on attention patterns.
    We'll use flattened JS divergence, then convert to similarity.

    Args:
        attention_weights: (batch_size, n_heads, seq_len, seq_len)

    Returns:
        similarity_matrix: (n_heads, n_heads) symmetric matrix [0, 1]
    """
    b, h, s1, s2 = attention_weights.shape

    # Average over batch to get a representative pattern
    avg_attn = np.mean(attention_weights, axis=0) # (h, s1, s2)

    # Flatten spatial dims to treat each head as one large distribution for comparison
    # (Note: it's not strictly a single distribution, but a collection of s1 distributions.
    # We'll normalize the whole thing to 1 for a rough divergence metric)
    flat_attn = avg_attn.reshape(h, -1)
    flat_attn = flat_attn / np.sum(flat_attn, axis=1, keepdims=True)

    sim_matrix = np.zeros((h, h))
    for i in range(h):
        for j in range(h):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif i < j:
                jsd = compute_js_divergence(flat_attn[i], flat_attn[j])
                # Convert divergence [0, ln(2)] to similarity [0, 1]
                # JS max is ln(2) ~ 0.693
                sim = max(0.0, 1.0 - (jsd / np.log(2)))
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

    return sim_matrix
