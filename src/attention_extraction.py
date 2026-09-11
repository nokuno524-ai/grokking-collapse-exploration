import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Iterator, Dict, List, Tuple
import math

@contextmanager
def extract_attention(model: nn.Module) -> Iterator[Dict[int, List[torch.Tensor]]]:
    """
    Context manager to extract multi-head attention weights from ModularArithmeticTransformer layers.
    It temporarily monkey-patches the self_attn.forward method to intercept and return attention weights.

    Args:
        model: An instance of ModularArithmeticTransformer or similar model with transformer layers.

    Yields:
        A dictionary mapping layer_idx (int) to a list of attention weight tensors.
        Each tensor has shape (batch, n_heads, seq_len, seq_len).
    """
    attn_weights: Dict[int, List[torch.Tensor]] = {}
    original_forwards = {}

    try:
        # We need to find the MultiheadAttention modules
        for layer_idx, layer in enumerate(model.transformer.layers):
            attn_module = layer.self_attn
            original_forward = attn_module.forward
            original_forwards[layer_idx] = (attn_module, original_forward)
            attn_weights[layer_idx] = []

            # Monkey-patch to force weights extraction
            def create_patched_forward(orig_fwd, l_idx):
                def new_forward(*args, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False

                    # Force SDPA to use math backend so it computes and returns weights
                    # instead of flash attention which doesn't support returning weights
                    if hasattr(torch.nn.attention, 'sdpa_kernel'):
                        # PyTorch >= 2.2
                        from torch.nn.attention import sdpa_kernel, SDPBackend
                        with sdpa_kernel([SDPBackend.MATH]):
                            out, weights = orig_fwd(*args, **kwargs)
                    else:
                        # PyTorch < 2.2 fallback
                        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                            out, weights = orig_fwd(*args, **kwargs)

                    attn_weights[l_idx].append(weights)
                    return out, weights
                return new_forward

            attn_module.forward = create_patched_forward(original_forward, layer_idx)

        yield attn_weights

    finally:
        # Restore original forward methods
        for layer_idx, (attn_module, original_forward) in original_forwards.items():
            attn_module.forward = original_forward


def compute_attention_entropy(attn_matrix: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute the entropy of attention distributions.

    Args:
        attn_matrix: Tensor of shape (..., seq_len, seq_len) containing attention probabilities.
        eps: Small value to avoid log(0).

    Returns:
        Entropy per head/batch of shape (..., seq_len).
    """
    # attn_matrix should sum to 1 over the last dimension
    entropy = -torch.sum(attn_matrix * torch.log(attn_matrix + eps), dim=-1)
    return entropy

def compute_diag_offdiag_ratio(attn_matrix: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Compute the ratio of diagonal mass to off-diagonal mass in the matrix.

    Args:
        attn_matrix: Tensor of shape (..., seq_len, seq_len).

    Returns:
        Ratio per head/batch of shape (...).
    """
    seq_len = attn_matrix.shape[-1]

    # Create mask for diagonal
    diag_mask = torch.eye(seq_len, dtype=torch.bool, device=attn_matrix.device)

    # Extract diagonal and off-diagonal masses
    expand_dims = list(attn_matrix.shape[:-2]) + [seq_len, seq_len]
    diag_mask_expanded = diag_mask.expand(*expand_dims)

    diag_mass = torch.sum(attn_matrix * diag_mask_expanded, dim=(-2, -1))
    offdiag_mass = torch.sum(attn_matrix * (~diag_mask_expanded), dim=(-2, -1))

    ratio = diag_mass / (offdiag_mass + eps)
    return ratio

def compute_head_specialization(attn_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute how specialized a head is by comparing its attention pattern to the mean pattern.

    Args:
        attn_matrix: Tensor of shape (batch, n_heads, seq_len, seq_len).

    Returns:
        Specialization index of shape (batch, n_heads).
    """
    mean_attn = torch.mean(attn_matrix, dim=1, keepdim=True)
    specialization = torch.mean(torch.abs(attn_matrix - mean_attn), dim=(-2, -1))
    return specialization

def extract_attention_metrics(model: nn.Module,
                              inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run forward pass and extract comprehensive attention metrics.

    Args:
        model: Transformer model.
        inputs: Input tensor (batch, seq_len).

    Returns:
        Tuple of (logits, dict of metrics).
    """
    metrics = {
        'weights': [],
        'entropy': [],
        'diag_ratio': [],
        'specialization': []
    }

    with extract_attention(model) as extracted:
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            logits = model(inputs)

        for layer_idx in sorted(extracted.keys()):
            if not extracted[layer_idx]:
                continue

            w = extracted[layer_idx][0]  # (batch, n_heads, seq_len, seq_len)

            metrics['weights'].append(w)
            metrics['entropy'].append(compute_attention_entropy(w))
            metrics['diag_ratio'].append(compute_diag_offdiag_ratio(w))
            metrics['specialization'].append(compute_head_specialization(w))

    if metrics['weights']:
        metrics['weights'] = torch.stack(metrics['weights'])
        metrics['entropy'] = torch.stack(metrics['entropy'])
        metrics['diag_ratio'] = torch.stack(metrics['diag_ratio'])
        metrics['specialization'] = torch.stack(metrics['specialization'])

    return logits, metrics