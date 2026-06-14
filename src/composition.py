"""
Compositional structure analysis utilities.
Measures how attention heads interact and compose.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import numpy as np

def analyze_composition(model: nn.Module, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Measure Q/K/V composition scores per pair of heads.
    Since this is a 1-layer transformer, inter-layer composition doesn't strictly exist
    in the typical Anthropic sense (Layer 1 head composing with Layer 2 head).
    However, we can measure how different heads within the same layer or across
    different positions/tokens interact, or if we generalize to multi-layer:
    measure how much the output of Head A is used by Q/K/V of Head B.

    Since the current model is 1-layer, we will calculate the dot products of
    WV, WQ, WK matrices to detect potential virtual head compositions or self-composition.
    """
    # Assuming ModularArithmeticTransformer structure
    encoder_layer = model.transformer.layers[0] if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers') else None
    if encoder_layer is None:
        return {}

    attn = encoder_layer.self_attn
    d_model = model.d_model
    n_heads = model.n_heads
    head_dim = d_model // n_heads

    # Extract weights
    # in_proj_weight is shape (3 * d_model, d_model) for Q, K, V
    if attn.in_proj_weight is not None:
        W_qkv = attn.in_proj_weight
        W_q, W_k, W_v = W_qkv.chunk(3, dim=0)
    else:
        # Fallback if using separate weights
        W_q = attn.q_proj_weight
        W_k = attn.k_proj_weight
        W_v = attn.v_proj_weight

    W_o = attn.out_proj.weight  # (d_model, d_model)

    # Reshape weights into per-head matrices
    # W_q: (n_heads * head_dim, d_model) -> (n_heads, head_dim, d_model)
    W_q_heads = W_q.view(n_heads, head_dim, d_model)
    W_k_heads = W_k.view(n_heads, head_dim, d_model)
    W_v_heads = W_v.view(n_heads, head_dim, d_model)

    # W_o: (d_model, n_heads * head_dim) -> (d_model, n_heads, head_dim)
    # Actually out_proj.weight is (d_model, d_model) where input is concat(heads)
    W_o_heads = W_o.view(d_model, n_heads, head_dim).permute(1, 0, 2)  # (n_heads, d_model, head_dim)

    composition_scores = {
        'q_composition': torch.zeros((n_heads, n_heads)),
        'k_composition': torch.zeros((n_heads, n_heads)),
        'v_composition': torch.zeros((n_heads, n_heads))
    }

    # In a 1-layer model, true composition from L1 to L2 doesn't exist.
    # We calculate the hypothetical score: if the output of Head A was fed into Head B.
    # Score = ||W_q_B * W_o_A||_F / (||W_q_B||_F * ||W_o_A||_F)

    for i in range(n_heads): # Head A (source)
        for j in range(n_heads): # Head B (dest)
            W_o_A = W_o_heads[i] # (d_model, head_dim)

            W_q_B = W_q_heads[j] # (head_dim, d_model)
            W_k_B = W_k_heads[j] # (head_dim, d_model)
            W_v_B = W_v_heads[j] # (head_dim, d_model)

            # Products
            OV_q = W_q_B @ W_o_A # (head_dim, head_dim)
            OV_k = W_k_B @ W_o_A
            OV_v = W_v_B @ W_o_A

            # Normalize
            norm_q = torch.norm(W_q_B) * torch.norm(W_o_A)
            norm_k = torch.norm(W_k_B) * torch.norm(W_o_A)
            norm_v = torch.norm(W_v_B) * torch.norm(W_o_A)

            composition_scores['q_composition'][i, j] = torch.norm(OV_q) / (norm_q + 1e-9)
            composition_scores['k_composition'][i, j] = torch.norm(OV_k) / (norm_k + 1e-9)
            composition_scores['v_composition'][i, j] = torch.norm(OV_v) / (norm_v + 1e-9)

    return composition_scores

def composition_matrix(model: nn.Module, dataloader) -> torch.Tensor:
    """
    Compute the full HxH composition matrix.
    Since we only have 1 layer, we compute an aggregated composition score
    representing the average structural coupling between heads.
    """
    # Use a dummy input to get scores
    dummy_input = torch.zeros(1, 2, dtype=torch.long)
    scores = analyze_composition(model, dummy_input)

    if not scores:
        return torch.zeros(0, 0)

    # Aggregate Q, K, V compositions into a single matrix
    total_comp = scores['q_composition'] + scores['k_composition'] + scores['v_composition']
    return total_comp / 3.0

def detect_circuits(comp_matrix: torch.Tensor, threshold: float = 0.1) -> List[Tuple[int, int]]:
    """
    Find strongly connected head clusters based on the composition matrix.

    Args:
        comp_matrix: HxH tensor of composition scores
        threshold: minimum score to be considered a connection

    Returns:
        List of (source_head, target_head) tuples
    """
    circuits = []
    if comp_matrix.numel() == 0:
        return circuits

    n_heads = comp_matrix.size(0)
    for i in range(n_heads):
        for j in range(n_heads):
            if comp_matrix[i, j].item() > threshold:
                circuits.append((i, j))

    return circuits

def track_composition_evolution(checkpoints: List[Tuple[int, nn.Module]]) -> Dict[int, torch.Tensor]:
    """
    Show how composition patterns change during training.

    Args:
        checkpoints: List of (step, model)

    Returns:
        Dictionary mapping step to its HxH composition matrix
    """
    evolution = {}
    for step, model in checkpoints:
        # Pass None for dataloader as our composition_matrix doesn't currently use it
        evolution[step] = composition_matrix(model, None)
    return evolution
