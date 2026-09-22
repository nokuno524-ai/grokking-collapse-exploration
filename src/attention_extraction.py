"""
Attention extraction context manager for modular arithmetic transformers.
"""

import torch
import torch.nn as nn
from typing import Generator, List, Any
from contextlib import contextmanager
import warnings


def compute_entropy(attn_weights: 'torch.Tensor | np.ndarray', eps: float = 1e-9) -> 'np.ndarray':
    """
    Compute Shannon entropy of attention weights along the key dimension.

    Args:
        attn_weights: Array of shape (..., heads, seq_len_q, seq_len_k)
        eps: Small value to prevent log(0)

    Returns:
        Array of shape (..., heads, seq_len_q) containing the entropy
    """
    import numpy as np
    attn = np.asarray(attn_weights)
    return -np.sum(attn * np.log(attn + eps), axis=-1)

@contextmanager
def extract_attention(model: nn.Module) -> Generator[List[torch.Tensor], None, None]:
    """
    Context manager to extract attention weights from a PyTorch transformer model.

    Temporarily patches nn.MultiheadAttention.forward to intercept kwargs
    (`need_weights=True`, `average_attn_weights=False`), forces eager fallback
    using torch.backends.cuda.sdp_kernel, and collects the `.detach().cpu().numpy()`
    arrays of the attention weights.

    Args:
        model: The transformer model to extract attention from.

    Yields:
        A list that will be populated with attention weight tensors (numpy arrays)
        during the forward pass. Each tensor has shape
        (batch_size, num_heads, target_seq_len, source_seq_len).
    """
    attentions: List[torch.Tensor] = []
    original_forward = nn.MultiheadAttention.forward

    def forward_with_attn(self: nn.MultiheadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False
        attn_output, attn_weights = original_forward(self, query, key, value, *args, **kwargs)
        if attn_weights is not None:
            attentions.append(attn_weights.detach().cpu().numpy())
        return attn_output, attn_weights

    # Patch the forward method
    nn.MultiheadAttention.forward = forward_with_attn # type: ignore

    # Disable fast path for TransformerEncoder in eval mode
    original_fast_paths = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.TransformerEncoder):
            # PyTorch 2.0+ uses a fast path in eval mode that bypasses MultiheadAttention
            original_fast_paths[name] = getattr(module, 'enable_nested_tensor', True)
            module.enable_nested_tensor = False

    try:
        # Suppress the deprecation warning for sdp_kernel which is expected
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # Force eager fallback to get weights, if SDPA is used
            try:
                # Use the new API if available
                from torch.nn.attention import sdpa_kernel, SDPBackend
                backends = [SDPBackend.MATH]
                cm = sdpa_kernel(backends)
            except ImportError:
                cm = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

            with cm:
                # To guarantee fast path is not taken, we can use a dummy context
                # where we make torch.is_grad_enabled() return True, but we still use torch.no_grad()
                # Actually, PyTorch checks model.training.
                # Let's just force the model.train() and disable dropout temporarily.
                original_training_state = {}
                original_dropout = {}
                for name, module in model.named_modules():
                    original_training_state[name] = module.training
                    module.training = True
                    if hasattr(module, 'dropout') and isinstance(module.dropout, float):
                        original_dropout[name] = ('float', module.dropout)
                        module.dropout = 0.0
                    elif isinstance(module, nn.Dropout):
                        original_dropout[name] = ('module', module.p)
                        module.p = 0.0

                try:
                    yield attentions
                finally:
                    # Restore training state
                    for name, module in model.named_modules():
                        module.training = original_training_state[name]
                        if name in original_dropout:
                            typ, val = original_dropout[name]
                            if typ == 'float':
                                module.dropout = val
                            else:
                                module.p = val

    finally:
        # Restore the original forward method
        nn.MultiheadAttention.forward = original_forward # type: ignore
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoder) and name in original_fast_paths:
                module.enable_nested_tensor = original_fast_paths[name]
