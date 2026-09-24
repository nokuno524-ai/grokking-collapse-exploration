"""
Utilities for extracting attention patterns from the ModularArithmeticTransformer.
"""

import contextlib
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Generator, List, Any


def compute_entropy(attn_weights: np.ndarray) -> float:
    """
    Compute Shannon entropy of an attention distribution.

    Args:
        attn_weights: 1D array of probabilities summing to 1.

    Returns:
        Entropy in nats.
    """
    # Filter out near-zero probabilities to avoid log(0)
    w = attn_weights[attn_weights > 1e-10]
    if len(w) == 0:
        return 0.0
    return -np.sum(w * np.log(w))


@contextlib.contextmanager
def extract_attention(model: nn.Module) -> Generator[List[np.ndarray], None, None]:
    """
    Context manager that temporarily patches nn.MultiheadAttention.forward
    to capture attention weights.

    Yields a list that will be populated with extracted attention map arrays
    (detached CPU numpy arrays of shape [batch_size, num_heads, seq_len, seq_len])
    after the forward pass completes.
    """
    extracted_maps: List[np.ndarray] = []

    # Store original forward methods and fastpath states
    original_forwards = {}
    original_fastpaths = {}

    # Locate all MultiheadAttention modules and TransformerEncoderLayers
    attn_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            attn_modules.append(module)
        if isinstance(module, nn.TransformerEncoderLayer):
            # Disable fastpath for SDPA so we can extract attention weights
            if hasattr(module.self_attn, 'fastpath'):
                original_fastpaths[module.self_attn] = module.self_attn.fastpath
                module.self_attn.fastpath = False

    for attn_module in attn_modules:
        original_forwards[attn_module] = attn_module.forward

        # Create a patched forward function closing over the current attn_module
        def make_patched_forward(orig_fwd):
            def patched_forward(*args: Any, **kwargs: Any) -> Any:
                # Force need_weights=True
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False  # Get per-head weights

                # Call original forward
                attn_output, attn_weights_tensor = orig_fwd(*args, **kwargs)

                # Store the weights
                if attn_weights_tensor is not None:
                    # Shape should be [batch_size, num_heads, seq_len, seq_len]
                    extracted_maps.append(attn_weights_tensor.detach().cpu().numpy())

                return attn_output, attn_weights_tensor
            return patched_forward

        attn_module.forward = make_patched_forward(original_forwards[attn_module])


    original_mha_fastpath = None
    if hasattr(torch, "backends") and hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        original_mha_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)

    try:
        yield extracted_maps
    finally:
        if original_mha_fastpath is not None:
            torch.backends.mha.set_fastpath_enabled(original_mha_fastpath)

        # Restore original forward methods and fastpaths
        for attn_module, orig_fwd in original_forwards.items():
            attn_module.forward = orig_fwd

        for self_attn, fastpath in original_fastpaths.items():
            self_attn.fastpath = fastpath
