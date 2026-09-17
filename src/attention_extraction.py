import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Generator, List
import numpy as np

@contextmanager
def extract_attention() -> Generator[List[np.ndarray], None, None]:
    """
    Context manager to extract attention weights from all nn.MultiheadAttention
    layers in a model during a forward pass.

    Temporarily patches nn.MultiheadAttention.forward to force eager execution
    (disabling fused SDP) and capture the attention weights before returning.

    Yields:
        A list that will be populated with attention weights of shape
        (batch_size, num_heads, seq_len, seq_len) for each layer.
    """
    attentions = []
    original_forward = nn.MultiheadAttention.forward

    def patched_forward(self, query, key, value, **kwargs):
        # Force eager execution to get attention weights
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False

        # We need to disable SDP backend momentarily to ensure need_weights works
        # This prevents 'fused' kernels from bypassing the explicit weights computation.
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            attn_output, attn_weights = original_forward(self, query, key, value, **kwargs)

        if attn_weights is not None:
            # We want shape: (batch_size, num_heads, seq_len, seq_len)
            # By default average_attn_weights=False returns this shape if num_heads > 1
            # Just to be safe, move it to CPU numpy
            attentions.append(attn_weights.detach().cpu().numpy())

        return attn_output, attn_weights

    # Apply patch
    nn.MultiheadAttention.forward = patched_forward

    try:
        yield attentions
    finally:
        # Restore original forward
        nn.MultiheadAttention.forward = original_forward
