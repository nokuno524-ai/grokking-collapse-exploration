"""
Mechanistic tools for analyzing attention head composition in Transformers.
"""

import torch
import torch.nn as nn
from typing import Dict


def compute_ov_composition(W_V: torch.Tensor, W_O: torch.Tensor) -> torch.Tensor:
    """
    Compute the composition of value and output weight matrices for an attention head.
    The OV circuit captures what information is moved to the residual stream once a
    token is attended to.

    Args:
        W_V: Value projection weights of shape (d_model, d_head) or (d_head, d_model)
        W_O: Output projection weights of shape (d_head, d_model) or (d_model, d_head)

    Returns:
        torch.Tensor: The composed OV matrix of shape (d_model, d_model)
    """
    # Assuming standard PyTorch Linear layer shapes: weight is (out_features, in_features)
    # W_V for a single head might need transposition depending on how it was extracted.
    # In PyTorch nn.MultiheadAttention, in_proj_weight is (3 * d_model, d_model).
    # This function expects W_V: (d_head, d_model) and W_O: (d_model, d_head)

    if W_V.shape[0] != W_O.shape[1]:
        # Try transposing if shapes mismatch
        W_O = W_O.t()

    # OV matrix is simply W_O @ W_V (if x is a column vector: W_O W_V x)
    return W_O @ W_V


def compute_qk_composition(W_Q: torch.Tensor, W_K: torch.Tensor) -> torch.Tensor:
    """
    Compute the composition of query and key weight matrices for an attention head.
    The QK circuit determines the attention pattern between tokens.

    Args:
        W_Q: Query projection weights
        W_K: Key projection weights

    Returns:
        torch.Tensor: The composed QK matrix.
    """
    # QK matrix is W_Q^T @ W_K  (or W_K^T @ W_Q depending on convention, typically x^T W_Q^T W_K y)
    # If standard linear layer: weight is (d_head, d_model)
    # Then query is W_Q x. Score is (W_Q x)^T (W_K y) = x^T W_Q^T W_K y
    return W_Q.t() @ W_K


def get_head_matrices(model: nn.Module, head_idx: int, layer_idx: int = 0) -> Dict[str, torch.Tensor]:
    """
    Extract Q, K, V, and O matrices for a specific attention head from a ModularArithmeticTransformer.

    Args:
        model: A trained ModularArithmeticTransformer model.
        head_idx: Index of the attention head to extract.
        layer_idx: Index of the transformer layer.

    Returns:
        Dict containing W_Q, W_K, W_V, W_O tensors for the specified head.
    """
    # We need to adapt based on ModularArithmeticTransformer's exact structure
    # It uses nn.TransformerEncoderLayer
    try:
        layer = model.transformer.layers[layer_idx]
        attn = layer.self_attn

        d_model = model.d_model
        n_heads = model.n_heads
        d_head = d_model // n_heads

        # in_proj_weight is shape (3 * d_model, d_model)
        if attn._qkv_same_embed_dim:
            qkv_weight = attn.in_proj_weight.detach()
            # Split into Q, K, V
            q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)

            # Extract specific head
            start = head_idx * d_head
            end = start + d_head

            W_Q = q_weight[start:end, :] # (d_head, d_model)
            W_K = k_weight[start:end, :] # (d_head, d_model)
            W_V = v_weight[start:end, :] # (d_head, d_model)

            # out_proj.weight is shape (d_model, d_model)
            W_O = attn.out_proj.weight.detach()[:, start:end] # (d_model, d_head)

            return {
                "W_Q": W_Q,
                "W_K": W_K,
                "W_V": W_V,
                "W_O": W_O
            }
        else:
            raise ValueError("Extraction not supported for differing QKV dims in this mock.")
    except Exception as e:
        # Fallback or empty if model structure doesn't match
        return {}
