import torch
import torch.nn as nn
from typing import Dict, Tuple

def get_weight_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute specific weight L2 norms for embedding, attention, and MLP.
    Returns a dictionary of norms.
    """
    norms = {
        "embedding": 0.0,
        "attention": 0.0,
        "mlp": 0.0,
        "output_head": 0.0
    }

    for name, param in model.named_parameters():
        norm_sq = param.norm().item() ** 2

        if "embed" in name:
            norms["embedding"] += norm_sq
        elif "transformer" in name and ("self_attn" in name or "attention" in name or "in_proj" in name or "out_proj" in name):
            norms["attention"] += norm_sq
        elif "transformer" in name and "linear" in name:
            # Typical TransformerEncoderLayer has linear1 and linear2 for FFN
            norms["mlp"] += norm_sq
        elif "output_head" in name:
            norms["output_head"] += norm_sq

    # Take square root to get true L2 norm
    for key in norms:
        norms[key] = norms[key] ** 0.5

    return norms

def effective_rank(matrix: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute effective rank of a matrix.
    Effective rank = exp(H(s)) where H is the Shannon entropy of normalized singular values.
    Includes safeguards against zero-tensors, NaNs, and infinities.
    """
    if matrix.numel() == 0:
        return 0.0

    # Handle zero tensors
    if torch.all(matrix == 0):
        return 0.0

    # Safeguard against NaN/Inf
    if not torch.isfinite(matrix).all():
        # Replace non-finite values with 0 for the purpose of rank calculation
        matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        s = torch.linalg.svdvals(matrix)
    except Exception:
        # SVD convergence failure or other errors
        return 0.0

    s_sum = s.sum()
    if s_sum <= eps:
        return 0.0

    s_norm = s / s_sum

    # Calculate entropy
    entropy = -(s_norm * torch.log(s_norm + eps)).sum()

    result = torch.exp(entropy).item()

    # Final safeguard
    if not torch.isfinite(torch.tensor(result)):
        return 0.0

    return result

def get_matrix_ranks(model: nn.Module) -> Dict[str, float]:
    """
    Compute effective rank for key weight matrices in the model.
    """
    ranks = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # For linear layers, matrix is out_features x in_features
            ranks[name] = effective_rank(module.weight.detach())
        elif isinstance(module, nn.Embedding):
            # For embeddings, matrix is num_embeddings x embedding_dim
            ranks[name] = effective_rank(module.weight.detach())

    # Also handle attention projection matrices if they are fused
    for name, param in model.named_parameters():
        if "in_proj_weight" in name:
            # Fused Q, K, V
            # Typically 3*d_model x d_model
            ranks[name] = effective_rank(param.detach())

    return ranks

def get_svd_distribution(matrix: torch.Tensor) -> torch.Tensor:
    """
    Extract singular values for tracking evolution.
    """
    if matrix.numel() == 0 or torch.all(matrix == 0):
        return torch.tensor([])

    if not torch.isfinite(matrix).all():
        matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        s = torch.linalg.svdvals(matrix)
        return s.detach()
    except Exception:
        return torch.tensor([])

def get_weight_histogram(matrix: torch.Tensor, bins: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute histogram of weight values.
    Returns (hist, bin_edges).
    """
    if matrix.numel() == 0:
        return torch.tensor([]), torch.tensor([])

    if not torch.isfinite(matrix).all():
        matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # PyTorch histogram function requires 1D tensor
    flat = matrix.flatten()

    # Ensure min < max
    min_val = flat.min().item()
    max_val = flat.max().item()

    if min_val >= max_val:
        min_val -= 1e-5
        max_val += 1e-5

    hist = torch.histc(flat, bins=bins, min=min_val, max=max_val)
    bin_edges = torch.linspace(min_val, max_val, bins + 1)

    return hist, bin_edges
