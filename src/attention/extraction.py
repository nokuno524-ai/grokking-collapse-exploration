import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager
from typing import List, Optional, Tuple, Dict, Any

def compute_entropy(attn_weights: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute Shannon entropy of the attention distributions.
    Args:
        attn_weights: (batch, n_heads, seq_len, seq_len)
        mask: Optional (batch, seq_len) where 1/True indicates valid tokens, 0/False indicates padding.
              (Note: sometimes mask semantics are reversed. Here we assume standard valid=True).
    Returns:
        entropy: (batch, n_heads, seq_len)
    """
    # Clamping to avoid log(0)
    Af = attn_weights.float().clamp_min(1e-12)

    if mask is not None:
        # mask is (batch, seq_len)
        # expand to (batch, 1, 1, seq_len) to mask out padded keys
        # Wait, if we mask out padded keys, we should renormalize the remaining probabilities
        # before computing entropy.
        k_mask = mask.bool().unsqueeze(1).unsqueeze(2)
        Af = Af * k_mask
        row_sums = Af.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        Af = Af / row_sums

    Af_clamped = Af.clamp_min(1e-12)
    H = -(Af_clamped * Af_clamped.log()).sum(dim=-1)

    # if we want to mask queries that are padding, we can zero out their entropy
    if mask is not None:
        q_mask = mask.bool().unsqueeze(1) # (batch, 1, seq_len)
        H = H * q_mask

    return H


@contextmanager
def extract_attention(model: nn.Module):
    """
    Context manager that temporarily patches nn.MultiheadAttention.forward
    to intercept calls and capture attention weights.
    Yields a list that will be populated with detached attention weights (numpy arrays).
    """
    attentions = []

    original_forwards = {}

    # Find all MultiheadAttention modules
    mha_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            mha_modules.append(module)

    def make_hooked_forward(orig_forward):
        def hooked_forward(*args, **kwargs):
            # Extract positional args manually to avoid signature issues
            # MultiheadAttention.forward(query, key, value, key_padding_mask=None,
            # need_weights=True, attn_mask=None, average_attn_weights=True, ...)

            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False

            out, attn = orig_forward(*args, **kwargs)

            if attn is not None:
                attentions.append(attn.detach().cpu().numpy())

            return out, attn
        return hooked_forward

    try:
        for module in mha_modules:
            original_forwards[module] = module.forward
            module.forward = make_hooked_forward(module.forward)

        # Passing need_weights=True prevents SDPA fast paths automatically,
        # so we do not need to mess with backends.
        yield attentions

    finally:
        for module in mha_modules:
            module.forward = original_forwards[module]


def compute_head_features(attn_weights: np.ndarray, tokens: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Compute attention head features for taxonomy:
    - entropy
    - positional bias (e.g. mass on offset -1, offset -2)
    - diagonal mass (mass on main diagonal)
    - copy score (mass on same tokens)
    - induction score (mass on tokens following same tokens)

    Args:
        attn_weights: (batch, n_heads, seq_len, seq_len) or list of such arrays
        tokens: (batch, seq_len)
        mask: Optional (batch, seq_len)

    Returns:
        Dict mapping feature names to (batch, n_heads) arrays.
    """
    if isinstance(attn_weights, list):
        # We might have multiple layers. Let's stack or assume single layer for now.
        # If multiple layers, assume shape (layers, batch, heads, seq, seq).
        # For simplicity, if list, just process the first one or stack.
        attn_weights = np.array(attn_weights)
        if attn_weights.ndim == 5:
            # (layers, batch, heads, seq, seq)
            # Flatten layers and heads into a single heads dimension
            L, B, H, S, S = attn_weights.shape
            attn_weights = attn_weights.transpose(1, 0, 2, 3, 4).reshape(B, L*H, S, S)

    B, H, S, _ = attn_weights.shape

    if mask is None:
        mask = np.ones((B, S), dtype=bool)

    # Valid queries
    q_mask = mask # (B, S)

    features = {}

    # 1. Entropy
    # Convert back to tensor for compute_entropy
    attn_t = torch.tensor(attn_weights)
    mask_t = torch.tensor(mask)
    entropy_t = compute_entropy(attn_t, mask_t) # (B, H, S)
    # Average over valid queries
    q_lengths = mask.sum(axis=1)[:, None] # (B, 1)
    q_lengths = np.maximum(q_lengths, 1)
    avg_entropy = entropy_t.numpy().sum(axis=-1) / q_lengths # (B, H)
    features['entropy'] = avg_entropy

    # 2. Diagonal mass (attention to current token)
    diag_indices = np.arange(S)
    diag_mass = attn_weights[:, :, diag_indices, diag_indices] # (B, H, S)
    diag_mass = diag_mass * mask[:, None, :] # mask out padded queries
    avg_diag_mass = diag_mass.sum(axis=-1) / q_lengths # (B, H)
    features['diagonal_mass'] = avg_diag_mass

    # 3. Positional bias (offset -1, offset -2)
    # offset -1: attending to previous token
    # offset -2: attending to two tokens back
    for offset in [1, 2]:
        mass = np.zeros((B, H, S))
        for i in range(offset, S):
            mass[:, :, i] = attn_weights[:, :, i, i - offset]
        mass = mass * mask[:, None, :]
        features[f'offset_{offset}_mass'] = mass.sum(axis=-1) / q_lengths

    # 4. Copy score
    # Attending to previous occurrences of the same token
    # (not counting the current token)
    copy_mass = np.zeros((B, H, S))
    for b in range(B):
        for i in range(1, S):
            for j in range(i):
                if tokens[b, i] == tokens[b, j]:
                    copy_mass[b, :, i] += attn_weights[b, :, i, j]
    copy_mass = copy_mass * mask[:, None, :]
    features['copy_score'] = copy_mass.sum(axis=-1) / q_lengths

    # 5. Induction score
    # Attending to a token that followed a previous occurrence of the current token.
    # i.e., at pos i, we have token T. We look at previous pos j where tokens[j] == tokens[i].
    # Then we check if we attend to j+1.
    induction_mass = np.zeros((B, H, S))
    for b in range(B):
        for i in range(2, S):
            # For each previous matching token
            for j in range(i - 1): # j is previous occurrence
                if tokens[b, i-1] == tokens[b, j]: # The context before current token matches
                    # Did we attend to the next token?
                    induction_mass[b, :, i] += attn_weights[b, :, i, j + 1]
    induction_mass = induction_mass * mask[:, None, :]
    features['induction_score'] = induction_mass.sum(axis=-1) / q_lengths

    return features
