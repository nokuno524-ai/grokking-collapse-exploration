import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, List, Optional
import numpy as np


@contextmanager
def extract_attention(model: nn.Module):
    """
    Context manager to extract attention matrices from nn.MultiheadAttention.

    Temporarily patches nn.MultiheadAttention.forward to intercept calls and
    extract the attention weights. It explicitly disables PyTorch 2.0+ SDPA
    fast-paths during eval mode to ensure hooks fire and attention matrices
    are computed and returned.

    Yields:
        A list that will be populated with extracted attention patterns.
        Each element is a dictionary mapping layer index to a numpy array
        of shape (batch_size, num_heads, seq_len, seq_len).
    """
    extracted = []
    handles = []

    # Force SDPA fastpath off to ensure attention weights can be captured
    original_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)

    # Keep track of layer indices
    layer_idx = 0

    # We will use a pre-hook or wrapper around forward.
    # We'll replace the forward method temporarily to intercept kwargs.

    original_forwards = {}

    def get_forward_wrapper(original_forward, idx):
        def forward_wrapper(*args, **kwargs):
            # Ensure need_weights=True and average_attn_weights=False
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False

            out, attn_weights = original_forward(*args, **kwargs)

            # attn_weights is (batch_size, num_heads, tgt_len, src_len)
            if attn_weights is not None:
                # Ensure it exists in the list
                if len(extracted) == 0:
                    extracted.append({})
                extracted[0][idx] = attn_weights.detach().cpu().numpy()

            return out, attn_weights
        return forward_wrapper

    try:
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                original_forwards[module] = module.forward
                module.forward = get_forward_wrapper(module.forward, layer_idx)
                layer_idx += 1

        yield extracted

    finally:
        torch.backends.mha.set_fastpath_enabled(original_fastpath)
        for module, orig_fwd in original_forwards.items():
            module.forward = orig_fwd

def compute_entropy(attention_matrix: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Shannon entropy of an attention matrix.

    Args:
        attention_matrix: numpy array of shape (..., seq_len)
        mask: optional boolean mask of valid keys (True for valid, False for padding)
              Shape should broadcast with attention_matrix.

    Returns:
        Mean entropy across all dimensions except the last one (seq_len).
    """
    A = attention_matrix

    if mask is not None:
        # Apply mask and re-normalize
        A = A * mask
        A_sum = np.sum(A, axis=-1, keepdims=True)
        # Avoid division by zero
        A_sum = np.maximum(A_sum, 1e-12)
        A = A / A_sum

    Af = np.maximum(A, 1e-12)
    H = -np.sum(Af * np.log(Af), axis=-1)

    if mask is not None:
        # Use a query mask if provided (usually we just mean over all valid queries)
        # For simplicity, we assume mask applies to keys.
        # If we want to strictly mask queries, we could do it here.
        # Assuming all valid tokens for now as it's typically a square mask for causal.
        pass

    return float(np.mean(H))
