import numpy as np

def attention_entropy(attn_matrix: np.ndarray) -> float:
    """
    Compute mean entropy of attention distributions across heads and queries.

    Args:
        attn_matrix: shape (batch_size, n_heads, seq_len, seq_len)

    Returns:
        Mean entropy (float).
    """
    eps = 1e-10
    # Calculate entropy per query: H = -sum(p * log(p)) over key dimension (last dim)
    entropy_per_query = -np.sum(attn_matrix * np.log(attn_matrix + eps), axis=-1) # (batch_size, n_heads, seq_len)

    # Average across batch, heads, and queries
    return float(np.mean(entropy_per_query))

def attention_entropy_per_head(attn_matrix: np.ndarray) -> np.ndarray:
    """
    Compute mean entropy of attention distributions per head.

    Args:
        attn_matrix: shape (batch_size, n_heads, seq_len, seq_len)

    Returns:
        Mean entropy per head, shape (n_heads,)
    """
    eps = 1e-10
    entropy_per_query = -np.sum(attn_matrix * np.log(attn_matrix + eps), axis=-1) # (batch_size, n_heads, seq_len)

    # Average across batch and queries
    return np.mean(entropy_per_query, axis=(0, 2))

def head_specialization_score(attn_matrix: np.ndarray) -> float:
    """
    Compute variance of attention over query positions.
    Higher variance indicates specialization (head attends differently depending on query).

    Args:
        attn_matrix: shape (batch_size, n_heads, seq_len, seq_len)

    Returns:
        Mean specialization score across heads (float)
    """
    # Variance of the attention distribution across queries (axis=2)
    # Average over batch, heads, and keys
    return float(np.mean(np.var(attn_matrix, axis=2)))

def head_specialization_score_per_head(attn_matrix: np.ndarray) -> np.ndarray:
    """
    Compute variance of attention over query positions per head.

    Args:
        attn_matrix: shape (batch_size, n_heads, seq_len, seq_len)

    Returns:
        Specialization score per head, shape (n_heads,)
    """
    # Variance across queries (dim=2), average across batch (dim=0) and keys (dim=3)
    return np.mean(np.var(attn_matrix, axis=2), axis=(0, 2))

def circuit_formation_index(attn_matrix_prev: np.ndarray, attn_matrix_curr: np.ndarray) -> float:
    """
    Compute change-rate of attention structure across checkpoints.
    Defined as the mean absolute difference between attention matrices.

    Args:
        attn_matrix_prev: shape (batch_size, n_heads, seq_len, seq_len)
        attn_matrix_curr: shape (batch_size, n_heads, seq_len, seq_len)

    Returns:
        Change rate (float)
    """
    return float(np.mean(np.abs(attn_matrix_curr - attn_matrix_prev)))
