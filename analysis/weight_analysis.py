import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import numpy as np

def estimate_hessian_top_eigenvalue(model: nn.Module, loss_fn, inputs: torch.Tensor, targets: torch.Tensor, num_iters: int = 10) -> float:
    """
    Approximate the top eigenvalue of the Hessian using power iteration.
    Optimized for k=1 usage per memory constraints (no full deflation).
    """
    model.eval()

    # Generate random vector v of same shape as parameters
    v = [torch.randn(p.size(), device=p.device) for p in model.parameters() if p.requires_grad]
    # Normalize v
    v_norm = torch.sqrt(sum(torch.sum(x**2) for x in v))
    v = [x / v_norm for x in v]

    for i in range(num_iters):
        model.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        # First derivative
        grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=True)

        # Hessian-vector product
        # grad * v
        g_v = sum(torch.sum(g * x) for g, x in zip(grads, v))

        # Second derivative
        Hv = torch.autograd.grad(g_v, [p for p in model.parameters() if p.requires_grad], retain_graph=True)

        # Compute Rayleigh quotient (v^T H v)
        rayleigh = sum(torch.sum(h * x).item() for h, x in zip(Hv, v))

        # Update v for next iteration
        Hv_norm = torch.sqrt(sum(torch.sum(h**2) for h in Hv))
        if Hv_norm == 0:
            return 0.0

        v = [h / Hv_norm for h in Hv]

    return rayleigh

def compute_effective_rank(W: torch.Tensor) -> float:
    """
    Compute effective rank as exp(H(s)) where H is Shannon entropy of normalized singular values.
    """
    s = torch.linalg.svdvals(W)
    s = s / (s.sum() + 1e-10)
    entropy = -(s * torch.log(s + 1e-10)).sum()
    return torch.exp(entropy).item()

def compute_compressibility_ratio(W: torch.Tensor) -> float:
    """
    Compute nuclear norm divided by Frobenius norm.
    """
    nuclear_norm = torch.linalg.norm(W, ord='nuc')
    frobenius_norm = torch.linalg.norm(W, ord='fro')
    if frobenius_norm < 1e-10:
        return 0.0
    return (nuclear_norm / frobenius_norm).item()

if __name__ == "__main__":
    W = torch.randn(10, 10)
    print("Effective rank:", compute_effective_rank(W))
    print("Compressibility ratio:", compute_compressibility_ratio(W))
