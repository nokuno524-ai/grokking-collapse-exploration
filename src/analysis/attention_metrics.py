import numpy as np

def compute_attention_entropy(attn_weights: np.ndarray) -> np.ndarray:
    """
    Compute row-wise entropy of attention weights.

    Args:
        attn_weights: numpy array of shape (..., tgt_len, src_len)
                      representing attention probabilities (sum to 1 over src_len).

    Returns:
        numpy array of shape (..., tgt_len) with the entropy of each row.
    """
    # Clip to avoid log(0)
    clipped = np.clip(attn_weights, 1e-10, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=-1)
    return entropy

def compute_diagonal_mass(attn_weights: np.ndarray) -> np.ndarray:
    """
    Compute the fraction of attention mass on the diagonal.

    Args:
        attn_weights: numpy array of shape (..., tgt_len, src_len)

    Returns:
        numpy array of shape (...) with the average diagonal mass.
    """
    tgt_len, src_len = attn_weights.shape[-2:]
    diag_mask = np.eye(tgt_len, src_len, dtype=bool)

    # mass on diagonal
    diag_mass = np.sum(attn_weights[..., diag_mask], axis=-1)
    # average over tgt_len
    return diag_mass / min(tgt_len, src_len)

def compute_head_specialization(attn_weights: np.ndarray) -> float:
    """
    Measure how specialized different heads are.
    A simple metric: the variance of the attention matrices across heads.

    Args:
        attn_weights: numpy array of shape (batch, num_heads, tgt_len, src_len)

    Returns:
        float representing the average variance across heads.
    """
    if len(attn_weights.shape) < 4:
        return 0.0 # Cannot compute if no head dimension

    # Compute variance across the head dimension (axis=1)
    var = np.var(attn_weights, axis=1) # (batch, tgt_len, src_len)
    return float(np.mean(var))
