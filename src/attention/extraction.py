"""
Utilities for extracting attention weights and computing attention entropy.
"""

import contextlib
from typing import List, Generator
import torch
import torch.nn as nn

@contextlib.contextmanager
def extract_attention(model: nn.Module) -> Generator[List[torch.Tensor], None, None]:
    """
    Context manager to extract attention patterns from a model during forward passes.

    Temporarily patches nn.MultiheadAttention.forward to intercept attention weights,
    disabling PyTorch 2.0+ SDPA fast-paths to ensure weights are returned.

    Yields:
        A list that will be populated with attention weight tensors
        (shape: [batch_size, num_heads, seq_len, seq_len]) for each attention layer.

    Example:
        >>> with extract_attention(model) as attn_weights:
        >>>     model(inputs)
        >>> # attn_weights now contains the attention probabilities
    """
    attention_weights: List[torch.Tensor] = []

    original_mha_forward = nn.MultiheadAttention.forward

    def hooked_forward(self, query, key, value, *args, **kwargs):
        # We must explicitly request attention weights
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False

        # Call the original forward
        attn_output, attn_weight = original_mha_forward(self, query, key, value, *args, **kwargs)

        # Save the detached, CPU attention weights
        if attn_weight is not None:
            attention_weights.append(attn_weight.detach().cpu())

        return attn_output, attn_weight

    # Disable fast path globally within the context
    original_fastpath_state = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)

    # Patch the method
    nn.MultiheadAttention.forward = hooked_forward

    try:
        yield attention_weights
    finally:
        # Restore original state
        nn.MultiheadAttention.forward = original_mha_forward
        torch.backends.mha.set_fastpath_enabled(original_fastpath_state)

def compute_entropy(attn_weights: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the entropy of attention weights.

    Args:
        attn_weights: Tensor of shape (batch, heads, seq_len, seq_len)
        mask: Optional boolean mask of shape (batch, seq_len) where True means valid token
              and False means padding.

    Returns:
        Tensor of shape (batch, heads, seq_len) containing entropy per query position.
    """
    if mask is not None:
        # Shape: (batch, 1, 1, seq_len)
        expanded_mask = mask.bool().unsqueeze(1).unsqueeze(2)
        # Zero out padding keys
        attn_weights = attn_weights * expanded_mask
        # Renormalize
        row_sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn_weights = attn_weights / row_sums

    # Compute entropy: -sum(p * log(p))
    # Add epsilon to prevent log(0)
    eps = 1e-9
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)

    if mask is not None:
        # Zero out entropy for padding queries
        # mask shape is (batch, seq_len), want (batch, 1, seq_len)
        query_mask = mask.bool().unsqueeze(1)
        entropy = entropy * query_mask

    return entropy
