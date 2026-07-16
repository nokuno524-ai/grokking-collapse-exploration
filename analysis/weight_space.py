import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import math

def compute_weight_metrics(model: nn.Module) -> Dict[str, float]:
    """
    Computes weight norm, spectral norm, and effective rank for the token embeddings.

    Args:
        model: ModularArithmeticTransformer model.

    Returns:
        Dictionary of computed metrics.
    """
    # Total weight norm (L2)
    weight_norm = model.get_weight_norm()

    # Effective rank and spectral norm for token embedding
    W = model.token_embed.weight.detach()
    s = torch.linalg.svdvals(W)

    spectral_norm = s[0].item()

    # Effective rank (exp of Shannon entropy of normalized singular values)
    s_norm = s / s.sum()
    # Mask out zeros to avoid log(0)
    s_norm = s_norm[s_norm > 1e-10]
    entropy = -(s_norm * torch.log(s_norm)).sum()
    effective_rank = torch.exp(entropy).item()

    return {
        "weight_norm": weight_norm,
        "spectral_norm": spectral_norm,
        "effective_rank": effective_rank
    }

def compute_hessian_eigenvalues(model: nn.Module, data_loader: torch.utils.data.DataLoader, criterion: nn.Module, top_k: int = 1, num_iters: int = 100, device: torch.device = torch.device("cpu")) -> List[float]:
    """
    Computes the top-k Hessian eigenvalues using power iteration with deflation.

    Args:
        model: ModularArithmeticTransformer model.
        data_loader: DataLoader for the dataset.
        criterion: Loss function.
        top_k: Number of top eigenvalues to compute.
        num_iters: Number of power iterations.
        device: Device to perform computations on.

    Returns:
        List of top-k eigenvalues.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)

    def hvp(v):
        """Hessian-vector product."""
        # Calculate loss
        total_loss = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss * x.size(0)
        total_loss /= len(data_loader.dataset)

        # First derivative
        grads = torch.autograd.grad(total_loss, params, create_graph=True, retain_graph=True)
        flat_grads = torch.cat([g.view(-1) for g in grads])

        # Second derivative (HVP)
        hvp_val = torch.autograd.grad(flat_grads, params, grad_outputs=v, retain_graph=True)
        return torch.cat([h.contiguous().view(-1) for h in hvp_val])

    eigenvalues = []
    eigenvectors = []

    for k in range(top_k):
        # Initialize random vector
        v = torch.randn(num_params, device=device)
        v = v / torch.norm(v)

        # Power iteration
        for _ in range(num_iters):
            # Compute HVP
            hv = hvp(v)

            # Deflation: orthogonalize against previously found eigenvectors
            for i in range(k):
                ev = eigenvectors[i]
                proj = torch.dot(hv, ev)
                hv = hv - proj * ev

            # Normalize
            v = hv / torch.norm(hv)

        # Compute eigenvalue (Rayleigh quotient)
        hv = hvp(v)
        for i in range(k):
            ev = eigenvectors[i]
            proj = torch.dot(hv, ev)
            hv = hv - proj * ev
        eigenvalue = torch.dot(v, hv).item()

        eigenvalues.append(eigenvalue)
        eigenvectors.append(v.detach())

    return eigenvalues

def measure_weight_sparsity(model: nn.Module, threshold: float = 1e-4) -> float:
    """
    Measures the fraction of weights with magnitude below a given threshold.

    Args:
        model: ModularArithmeticTransformer model.
        threshold: Sparsity threshold.

    Returns:
        Fraction of sparse weights.
    """
    total_params = 0
    sparse_params = 0

    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
            sparse_params += (p.abs() < threshold).sum().item()

    if total_params == 0:
        return 0.0

    return sparse_params / total_params

def correlate_metrics(metrics1: List[float], metrics2: List[float]) -> float:
    """
    Computes Pearson correlation coefficient between two lists of metrics.

    Args:
        metrics1: First list of metrics.
        metrics2: Second list of metrics.

    Returns:
        Pearson correlation coefficient.
    """
    if len(metrics1) != len(metrics2) or len(metrics1) < 2:
        return 0.0

    t1 = torch.tensor(metrics1)
    t2 = torch.tensor(metrics2)

    # Pearson correlation
    vx = t1 - t1.mean()
    vy = t2 - t2.mean()

    # Avoid division by zero
    if torch.norm(vx) == 0 or torch.norm(vy) == 0:
         return 0.0

    r = torch.sum(vx * vy) / (torch.norm(vx) * torch.norm(vy))
    return r.item()
