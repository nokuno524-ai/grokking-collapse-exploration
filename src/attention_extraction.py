import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Generator, List, Dict, Callable
import numpy as np


@contextmanager
def extract_attention(model: nn.Module) -> Generator[List[np.ndarray], None, None]:
    """
    Context manager to extract attention weights from a PyTorch model containing
    nn.MultiheadAttention modules (like nn.TransformerEncoder).

    Yields a list that will be populated with attention weights.
    The weights are stored as numpy arrays of shape (batch, n_heads, seq_len, seq_len)
    to avoid storing PyTorch tensors and causing memory leaks.
    """
    attention_weights: List[np.ndarray] = []
    original_forwards: Dict[str, Callable] = {}

    def get_forward_with_weights(original_forward: Callable) -> Callable:
        def forward(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False # per-head attention
            out, weights = original_forward(*args, **kwargs)
            # Detach, move to CPU, and convert to numpy immediately to free PyTorch graphs/tensors
            attention_weights.append(weights.detach().cpu().numpy())
            return out, weights
        return forward

    # Replace all MultiheadAttention forwards with our wrapper
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            original_forwards[name] = module.forward
            module.forward = get_forward_with_weights(module.forward)

    try:
        # Force eager fallback for SDPA so that attention weights are materialized and returned
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            yield attention_weights
    finally:
        # Restore original forward methods
        for name, module in model.named_modules():
            if name in original_forwards:
                module.forward = original_forwards[name]
