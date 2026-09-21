import contextlib
import types
from typing import Any, Dict, Generator, List, Tuple
import numpy as np
import torch
import torch.nn as nn

@contextlib.contextmanager
def extract_attention(model: nn.Module) -> Generator[Dict[str, List[np.ndarray]], None, None]:
    """
    Context manager to extract attention patterns from a model's nn.MultiheadAttention modules.

    It temporarily patches the `forward` method of all `nn.MultiheadAttention` layers to intercept
    the returned attention weights. It also uses `torch.backends.cuda.sdp_kernel` to disable flash
    attention (which does not return attention weights) and forces an eager fallback.

    Yields:
        A dictionary mapping layer names (or indices if names are empty) to a list of attention
        weight matrices (as detached CPU numpy arrays).
        The matrices typically have shape (batch_size, num_heads, tgt_len, src_len).
    """
    extracted_weights: Dict[str, List[np.ndarray]] = {}
    original_forwards = {}

    def get_patched_forward(name: str, original_forward: Any):
        def patched_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs):
            # Intercept and force kwargs to return attention weights without averaging
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            attn_output, attn_weights = original_forward(query, key, value, **kwargs)

            # Store the attention weights as detached CPU numpy arrays
            if name not in extracted_weights:
                extracted_weights[name] = []
            extracted_weights[name].append(attn_weights.detach().cpu().numpy())

            return attn_output, attn_weights
        return patched_forward

    # Patch all MultiheadAttention modules
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            if not name:
                name = "layer_0" # fallback if root is multiheadattention
            original_forwards[name] = module.forward
            # Bind the patched forward to the module instance
            module.forward = types.MethodType(get_patched_forward(name, module.forward), module)

    # Disable flash/mem_efficient attention to force eager evaluation (which returns weights)
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            yield extracted_weights
    finally:
        # Restore original forward methods
        for name, module in model.named_modules():
            if name in original_forwards:
                module.forward = original_forwards[name]
