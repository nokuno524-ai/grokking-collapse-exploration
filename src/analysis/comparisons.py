import torch
import torch.nn as nn
from typing import Dict, List, Tuple

def cka_similarity(activations_a: torch.Tensor, activations_b: torch.Tensor) -> float:
    """
    Compute Linear CKA (Centred Kernel Alignment) between two sets of activations.
    Activations should be 2D tensors of shape (num_samples, num_features).
    """
    # Center the columns (features)
    a_centered = activations_a - activations_a.mean(dim=0, keepdim=True)
    b_centered = activations_b - activations_b.mean(dim=0, keepdim=True)

    # Compute dot product similarities
    dot_prod_a = torch.matmul(a_centered, a_centered.t())
    dot_prod_b = torch.matmul(b_centered, b_centered.t())

    # Frobenius inner product
    cka = torch.sum(dot_prod_a * dot_prod_b) / (torch.norm(dot_prod_a, p='fro') * torch.norm(dot_prod_b, p='fro'))
    return cka.item()

def svcca_similarity(activations_a: torch.Tensor, activations_b: torch.Tensor, retain_var: float = 0.99) -> float:
    """
    Compute SVCCA (Singular Vector Canonical Correlation Analysis) between two sets of activations.
    Activations should be 2D tensors of shape (num_samples, num_features).
    """
    # Center the activations
    a = activations_a - activations_a.mean(dim=0, keepdim=True)
    b = activations_b - activations_b.mean(dim=0, keepdim=True)

    # SVD
    U_a, S_a, V_a = torch.linalg.svd(a, full_matrices=False)
    U_b, S_b, V_b = torch.linalg.svd(b, full_matrices=False)

    # Calculate how many singular values to retain
    var_a = (S_a ** 2).cumsum(dim=0) / (S_a ** 2).sum()
    var_b = (S_b ** 2).cumsum(dim=0) / (S_b ** 2).sum()

    # Add small epsilon to avoid searchsorted issues if exact 1.0
    k_a = torch.searchsorted(var_a, retain_var).item() + 1
    k_b = torch.searchsorted(var_b, retain_var).item() + 1

    # Truncate
    U_a_trunc = U_a[:, :k_a]
    U_b_trunc = U_b[:, :k_b]

    # Compute cross-covariance matrix
    cross_cov = torch.matmul(U_a_trunc.t(), U_b_trunc)

    # SVD of cross-covariance to get CCA coefficients
    try:
        _, S, _ = torch.linalg.svd(cross_cov)
        return S.mean().item()
    except Exception as e:
        # Fallback if SVD fails to converge
        return 0.0

def specialization_score(model: nn.Module, dataset: torch.utils.data.Dataset) -> float:
    """
    Measure how specialized the attention heads are.
    We compute the entropy of the attention probabilities across heads.
    A higher entropy means heads are attending to the same things (less specialized).
    A lower entropy means heads have specialized attention patterns.
    We return (1.0 - normalized_entropy) as the specialization score (0 to 1).
    """
    # Note: ModularArithmeticTransformer uses nn.TransformerEncoderLayer which doesn't easily expose attn weights.
    # We will compute the empirical standard deviation of the out_proj weights grouped by head as a proxy.
    # Highly specialized models often have a few heads with very large weight norms.

    out_proj_weight = model.transformer.layers[0].self_attn.out_proj.weight.detach()
    n_heads = model.n_heads
    d_model = model.d_model
    head_dim = d_model // n_heads

    # (d_model, d_model) -> group input dims by head
    head_norms = torch.norm(out_proj_weight.view(d_model, n_heads, head_dim), dim=(0, 2))

    # Normalize to form a probability distribution
    p = head_norms / (head_norms.sum() + 1e-10)

    # Compute entropy
    entropy = -torch.sum(p * torch.log(p + 1e-10))

    # Max possible entropy is log(n_heads)
    max_entropy = torch.log(torch.tensor(n_heads, dtype=torch.float32))

    normalized_entropy = entropy / max_entropy
    return 1.0 - normalized_entropy.item()
