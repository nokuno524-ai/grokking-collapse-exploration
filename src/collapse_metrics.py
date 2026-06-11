import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np

def representation_collapse_score(representations: torch.Tensor, epsilon: float = 1e-10) -> float:
    """
    Measure the effective rank of hidden representations using Singular Value Decomposition.
    A lower score indicates more severe collapse (representations lie in a lower-dimensional subspace).

    Args:
        representations: Tensor of shape (batch_size, hidden_dim)
        epsilon: Small value to prevent log(0)

    Returns:
        Effective rank (entropy of singular values), typically in range [1.0, hidden_dim]
    """
    if len(representations.shape) > 2:
        # Flatten all dimensions except the last one (hidden_dim)
        representations = representations.reshape(-1, representations.shape[-1])

    # Subtract mean for PCA-like centering
    reps_centered = representations - representations.mean(dim=0, keepdim=True)

    # Compute singular values
    # Try SVD directly
    try:
        _, s, _ = torch.linalg.svd(reps_centered, full_matrices=False)
    except Exception:
        # Fallback if SVD fails to converge
        # SVD of X is related to eigenvalues of X^T X
        cov = torch.matmul(reps_centered.t(), reps_centered)
        eigvals = torch.linalg.eigvalsh(cov)
        # Eigenvalues are squared singular values, remove negative ones from numerical errors
        s = torch.sqrt(torch.clamp(eigvals, min=0.0))

    # Normalize to form a probability distribution
    s_norm = s / (s.sum() + epsilon)

    # Shannon entropy of normalized singular values
    entropy = -(s_norm * torch.log(s_norm + epsilon)).sum()

    # Exponentiated entropy gives effective rank
    effective_rank = torch.exp(entropy).item()
    return effective_rank


def gradient_collapse_score(gradients: List[torch.Tensor]) -> float:
    """
    Track gradient cosine similarity across training samples.
    A higher similarity score indicates gradients are aligning, potentially due to collapse.

    Args:
        gradients: List of 1D tensors (flattened gradients from different samples/batches)

    Returns:
        Average pairwise cosine similarity in range [-1.0, 1.0]
    """
    if len(gradients) < 2:
        return 0.0

    # Stack gradients into a matrix (N, D)
    grad_matrix = torch.stack([g.flatten() for g in gradients])

    # Normalize each gradient vector
    norms = torch.norm(grad_matrix, p=2, dim=1, keepdim=True)
    # Avoid division by zero
    norms = torch.clamp(norms, min=1e-10)
    normalized_grads = grad_matrix / norms

    # Compute pairwise cosine similarity matrix
    sim_matrix = torch.matmul(normalized_grads, normalized_grads.t())

    # Extract upper triangular part (excluding diagonal)
    # Get indices for upper triangular part
    n = len(gradients)
    triu_indices = torch.triu_indices(n, n, offset=1)

    # Calculate mean pairwise similarity
    mean_sim = sim_matrix[triu_indices[0], triu_indices[1]].mean().item()
    return mean_sim


def output_diversity_index(logits: torch.Tensor, epsilon: float = 1e-10) -> float:
    """
    Entropy of the output distribution across different inputs.
    Lower entropy indicates the model predicts a narrower set of outputs (output collapse).

    Args:
        logits: Output logits of shape (batch_size, num_classes)

    Returns:
        Entropy of the marginal class probability distribution
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # (batch_size, num_classes)

    # Compute marginal distribution across the batch
    marginal_probs = probs.mean(dim=0)  # (num_classes,)

    # Calculate entropy
    entropy = -(marginal_probs * torch.log(marginal_probs + epsilon)).sum()
    return entropy.item()


def weight_matrix_conditioning(weight_matrix: torch.Tensor) -> float:
    """
    Track the condition number of weight matrices.
    High condition number indicates numerical instability and potential collapse.

    Args:
        weight_matrix: 2D weight matrix (out_features, in_features)

    Returns:
        Condition number (ratio of largest to smallest singular value)
    """
    if len(weight_matrix.shape) != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {weight_matrix.shape}")

    try:
        s = torch.linalg.svdvals(weight_matrix)
    except Exception:
        # Fallback if SVD fails to converge
        cov = torch.matmul(weight_matrix.t(), weight_matrix)
        if weight_matrix.shape[0] < weight_matrix.shape[1]:
            cov = torch.matmul(weight_matrix, weight_matrix.t())

        eigvals = torch.linalg.eigvalsh(cov)
        s = torch.sqrt(torch.clamp(eigvals, min=0.0))

    max_sv = s.max().item()
    min_sv = s.min().item()

    # Avoid division by zero
    if min_sv < 1e-10:
        return float('inf')

    return max_sv / min_sv


def attention_pattern_collapse(attention_weights_list: List[torch.Tensor]) -> float:
    """
    Measure how attention patterns converge across different inputs.
    Higher values indicate attention patterns look the same regardless of input (collapse).

    Args:
        attention_weights_list: List of attention matrices, each (batch_size, num_heads, seq_len, seq_len)
                                or (num_heads, seq_len, seq_len)

    Returns:
        Average pairwise Frobenius distance or similar metric normalized to [0, 1]
        (1.0 means complete collapse/identical patterns)
    """
    if len(attention_weights_list) < 2:
        return 0.0

    # Flatten attention matrices for comparison
    flat_attns = []
    for attn in attention_weights_list:
        # If it has batch dimension, average over batch
        if len(attn.shape) == 4:
            attn = attn.mean(dim=0)
        flat_attns.append(attn.flatten())

    attn_matrix = torch.stack(flat_attns)  # (N, num_heads * seq_len * seq_len)

    # Calculate variance across different inputs for each attention weight
    # If patterns are identical (collapse), variance is 0
    # Normalize by max possible variance for probabilities (p * (1-p) <= 0.25)

    # Calculate standard deviation across inputs for each weight
    std_across_inputs = torch.std(attn_matrix, dim=0)

    # Calculate mean std across all weights
    mean_std = std_across_inputs.mean().item()

    # Inverse map so higher = more collapsed
    # Typical std for diverse attention might be 0.1-0.2
    # We use an exponential decay to map to [0, 1]
    collapse_score = math.exp(-10.0 * mean_std)
    return collapse_score
