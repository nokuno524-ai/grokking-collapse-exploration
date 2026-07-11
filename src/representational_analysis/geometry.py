import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

def compute_rsm(activations: torch.Tensor, metric: str = 'cosine') -> torch.Tensor:
    """
    Compute Representational Similarity Matrix (RSM) for a set of activations.

    Args:
        activations: Tensor of shape (n_samples, hidden_dim)
        metric: Distance metric to use ('cosine', 'euclidean', 'correlation')

    Returns:
        RSM tensor of shape (n_samples, n_samples)
    """
    if len(activations.shape) != 2:
        activations = activations.view(activations.size(0), -1)

    n_samples = activations.shape[0]

    if metric == 'cosine':
        # Normalize activations
        activations_norm = F.normalize(activations, p=2, dim=1)
        rsm = torch.mm(activations_norm, activations_norm.t())
    elif metric == 'euclidean':
        dists = torch.cdist(activations, activations, p=2)
        # Convert distance to similarity
        rsm = 1.0 / (1.0 + dists)
    elif metric == 'correlation':
        activations_np = activations.detach().cpu().numpy()
        dists = squareform(pdist(activations_np, metric='correlation'))
        rsm = torch.tensor(1.0 - dists, device=activations.device)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return rsm

def rsm_similarity(rsm1: torch.Tensor, rsm2: torch.Tensor) -> float:
    """
    Compute similarity between two RSMs using Pearson correlation of upper triangles.
    """
    assert rsm1.shape == rsm2.shape, "RSMs must have the same shape"

    # Get upper triangle elements, excluding diagonal
    triu_indices = torch.triu_indices(rsm1.shape[0], rsm1.shape[1], offset=1)

    vec1 = rsm1[triu_indices[0], triu_indices[1]].detach().cpu().numpy()
    vec2 = rsm2[triu_indices[0], triu_indices[1]].detach().cpu().numpy()

    corr, _ = pearsonr(vec1, vec2)
    return float(corr)

def compute_participation_ratio(activations: torch.Tensor) -> float:
    """
    Compute Participation Ratio (PR) as a measure of intrinsic dimensionality.
    PR = (tr(C))^2 / tr(C^2) where C is the covariance matrix.

    Args:
        activations: Tensor of shape (n_samples, hidden_dim)

    Returns:
        float: Participation Ratio
    """
    if len(activations.shape) != 2:
        activations = activations.view(activations.size(0), -1)

    # Center activations
    activations_centered = activations - activations.mean(dim=0, keepdim=True)

    # Compute covariance matrix (or Gram matrix if n_samples < hidden_dim to be efficient)
    n, d = activations_centered.shape

    if d <= n:
        cov = torch.mm(activations_centered.t(), activations_centered) / (n - 1)
        eigs = torch.linalg.eigvalsh(cov)
    else:
        # Dual trick: non-zero eigenvalues of X^T X are same as X X^T
        gram = torch.mm(activations_centered, activations_centered.t()) / (n - 1)
        eigs = torch.linalg.eigvalsh(gram)

    eigs = eigs[eigs > 0]

    if len(eigs) == 0:
        return 0.0

    pr = (torch.sum(eigs) ** 2) / torch.sum(eigs ** 2)
    return pr.item()

def compute_mle_id(activations: torch.Tensor, k: int = 10) -> float:
    """
    Compute Maximum Likelihood Estimation of Intrinsic Dimensionality (Levina & Bickel).

    Args:
        activations: Tensor of shape (n_samples, hidden_dim)
        k: Number of nearest neighbors

    Returns:
        float: MLE Intrinsic Dimensionality
    """
    if len(activations.shape) != 2:
        activations = activations.view(activations.size(0), -1)

    n = activations.shape[0]

    if n <= k:
        k = n - 1
        if k < 2:
            return 0.0

    # Compute pairwise Euclidean distances
    dists = torch.cdist(activations, activations, p=2)

    # Sort distances
    sorted_dists, _ = torch.sort(dists, dim=1)

    # Get distances to k-th neighbor
    k_dists = sorted_dists[:, k]

    # Filter out points with zero distance to k-th neighbor
    valid_mask = k_dists > 1e-8

    if not torch.any(valid_mask):
        return 0.0

    sorted_dists = sorted_dists[valid_mask]
    k_dists = k_dists[valid_mask]

    # Compute MLE ID for each valid point
    # \hat{m}_i = [ (k-1)^{-1} \sum_{j=1}^{k-1} \log(T_k(X_i) / T_j(X_i)) ]^{-1}
    log_dists = torch.log(k_dists.unsqueeze(1) / (sorted_dists[:, 1:k] + 1e-10))
    id_estimates = (k - 1) / torch.sum(log_dists, dim=1)

    return id_estimates.mean().item()

def compute_neural_anisotropy(activations: torch.Tensor, task_directions: torch.Tensor) -> float:
    """
    Compute neural anisotropy (alignment of representations with task-relevant directions).

    Args:
        activations: Tensor of shape (n_samples, hidden_dim)
        task_directions: Tensor of shape (n_directions, hidden_dim)

    Returns:
        float: Alignment score (mean squared cosine similarity)
    """
    if len(activations.shape) != 2:
        activations = activations.view(activations.size(0), -1)

    # Normalize
    act_norm = F.normalize(activations, p=2, dim=1)
    dir_norm = F.normalize(task_directions, p=2, dim=1)

    # Cosine similarities
    cos_sims = torch.mm(act_norm, dir_norm.t())

    # Mean squared cosine similarity across samples and directions
    anisotropy = torch.mean(cos_sims ** 2).item()
    return anisotropy

def compute_cka(activations_x: torch.Tensor, activations_y: torch.Tensor) -> float:
    """
    Compute linear Centered Kernel Alignment (CKA) between two sets of activations.

    Args:
        activations_x: Tensor of shape (n_samples, d1)
        activations_y: Tensor of shape (n_samples, d2)

    Returns:
        float: CKA score [0, 1]
    """
    if len(activations_x.shape) != 2:
        activations_x = activations_x.view(activations_x.size(0), -1)
    if len(activations_y.shape) != 2:
        activations_y = activations_y.view(activations_y.size(0), -1)

    assert activations_x.shape[0] == activations_y.shape[0], "Must have same number of samples"

    # Center columns
    x_centered = activations_x - activations_x.mean(dim=0, keepdim=True)
    y_centered = activations_y - activations_y.mean(dim=0, keepdim=True)

    # Dot products
    dot_prod = torch.sum(torch.mm(x_centered.t(), y_centered) ** 2)
    norm_x = torch.sum(torch.mm(x_centered.t(), x_centered) ** 2)
    norm_y = torch.sum(torch.mm(y_centered.t(), y_centered) ** 2)

    if norm_x == 0 or norm_y == 0:
        return 0.0

    cka = dot_prod / torch.sqrt(norm_x * norm_y)
    return cka.item()
