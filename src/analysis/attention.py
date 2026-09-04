"""
Quantitative analysis of attention patterns for the ModularArithmeticTransformer.
Includes extracting attention weights and computing metrics like entropy,
positional concentration, and head similarity.
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List

# Type alias
AttentionMap = torch.Tensor  # shape: (n_layers, n_heads, seq_len, seq_len)


@contextlib.contextmanager
def patch_attention_for_weights(model: nn.Module):
    """
    Context manager to extract attention weights from a ModularArithmeticTransformer.
    Temporarily patches the inner `self_attn.forward` to return attention weights,
    and forces the use of eager SDPA (MATH backend) to avoid issues with optimized kernels
    not returning weights.
    """
    # Find all MultiheadAttention modules
    mha_modules = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]

    original_forwards = {}

    for idx, mha in enumerate(mha_modules):
        original_forwards[idx] = mha.forward

        # Define a closure that captures the original forward but overrides kwargs
        def create_patched_forward(orig_fwd):
            def patched_forward(query, key, value, *args, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False
                return orig_fwd(query, key, value, *args, **kwargs)
            return patched_forward

        mha.forward = create_patched_forward(mha.forward)

    try:
        # Force the math backend for scaled_dot_product_attention if torch >= 2.0
        # Use sdpa_kernel if available (torch 2.1+), otherwise fallback to sdp_kernel
        if hasattr(torch.nn.attention, "sdpa_kernel"):
            from torch.nn.attention import SDPBackend
            with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):
                yield
        else:
            with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                yield
    finally:
        # Restore original forwards
        for idx, mha in enumerate(mha_modules):
            mha.forward = original_forwards[idx]


def extract_attention_weights(model: nn.Module, inputs: torch.Tensor) -> AttentionMap:
    """
    Extract attention weights from the model for a given batch of inputs.

    Args:
        model: ModularArithmeticTransformer instance
        inputs: Input tensor of shape (batch, seq_len)

    Returns:
        Attention weights tensor of shape (n_layers, n_heads, seq_len, seq_len)
        Averaged over the batch dimension.
    """
    # Hooks to capture output of MultiheadAttention
    attn_weights = []

    def hook(module, args, kwargs, output):
        # output of MHA with need_weights=True is (attn_output, attn_weights)
        # attn_weights shape: (batch, n_heads, seq_len, seq_len)
        if isinstance(output, tuple) and len(output) == 2:
            _, weights = output
            attn_weights.append(weights.detach())

    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(hook, with_kwargs=True))

    try:
        with patch_attention_for_weights(model):
            # Disable grad for extraction
            with torch.no_grad():
                _ = model(inputs)
    finally:
        for h in hooks:
            h.remove()

    if not attn_weights:
        raise ValueError("Could not extract attention weights. Are there any MultiheadAttention layers?")

    # Stack layers: shape becomes (n_layers, batch, n_heads, seq_len, seq_len)
    stacked = torch.stack(attn_weights, dim=0)

    # Average over batch: shape becomes (n_layers, n_heads, seq_len, seq_len)
    avg_weights = stacked.mean(dim=1)
    return avg_weights


def compute_entropy(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of attention distributions.

    Args:
        attn: Attention weights of shape (..., seq_len, seq_len)

    Returns:
        Entropy tensor of shape (..., seq_len)
    """
    # Add epsilon to prevent log(0)
    attn = torch.clamp(attn, min=1e-10)
    entropy = -(attn * torch.log(attn)).sum(dim=-1)
    return entropy


def compute_positional_concentration(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute positional mass concentration (max probability assigned to any single token).

    Args:
        attn: Attention weights of shape (..., seq_len, seq_len)

    Returns:
        Concentration tensor of shape (..., seq_len)
    """
    # Max probability over the key sequence dimension
    return attn.max(dim=-1).values


def compute_head_similarity(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between flattened attention maps of different heads.

    Args:
        attn: Attention weights of shape (n_layers, n_heads, seq_len, seq_len)

    Returns:
        Similarity matrix of shape (n_layers * n_heads, n_layers * n_heads)
    """
    n_layers, n_heads, seq_len, _ = attn.shape
    total_heads = n_layers * n_heads

    # Flatten: (n_layers * n_heads, seq_len * seq_len)
    flattened = attn.reshape(total_heads, seq_len * seq_len)

    # Compute cosine similarity
    # normalized = flattened / norm, sim = normalized @ normalized.T
    norms = flattened.norm(dim=-1, keepdim=True)
    norms = torch.clamp(norms, min=1e-10)
    normalized = flattened / norms

    similarity = normalized @ normalized.T
    return similarity


def analyze_attention(attn: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive metrics for an attention map.

    Args:
        attn: Attention weights of shape (n_layers, n_heads, seq_len, seq_len)

    Returns:
        Dictionary containing metric summaries as numpy arrays.
        - entropy: (n_layers, n_heads, seq_len)
        - mean_entropy: (n_layers, n_heads) - averaged over seq_len
        - concentration: (n_layers, n_heads, seq_len)
        - mean_concentration: (n_layers, n_heads)
        - head_similarity: (n_layers * n_heads, n_layers * n_heads)
    """
    entropy = compute_entropy(attn)
    concentration = compute_positional_concentration(attn)
    similarity = compute_head_similarity(attn)

    return {
        "entropy": entropy.cpu().numpy(),
        "mean_entropy": entropy.mean(dim=-1).cpu().numpy(),
        "concentration": concentration.cpu().numpy(),
        "mean_concentration": concentration.mean(dim=-1).cpu().numpy(),
        "head_similarity": similarity.cpu().numpy(),
    }
