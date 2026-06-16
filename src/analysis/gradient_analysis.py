"""
Mechanistic tools for analyzing gradient dynamics during training.
"""

import torch
import numpy as np
from typing import List


def compute_gradient_noise_scale(
    batch_grad: torch.Tensor,
    full_grad: torch.Tensor,
    batch_size: int
) -> float:
    """
    Compute the gradient noise scale, an SGLD-inspired metric that estimates
    the scale at which gradient noise dominates the true gradient signal.

    Args:
        batch_grad: Gradient computed on a mini-batch.
        full_grad: Gradient computed on the full dataset (or a very large batch).
        batch_size: The size of the mini-batch used for batch_grad.

    Returns:
        float: The estimated gradient noise scale.
    """
    # GNS is roughly |batch_grad - full_grad|^2 / |full_grad|^2 * batch_size
    diff = batch_grad - full_grad
    diff_norm_sq = torch.norm(diff) ** 2
    full_norm_sq = torch.norm(full_grad) ** 2

    if full_norm_sq < 1e-10:
        return 0.0

    noise_scale = (diff_norm_sq / full_norm_sq) * batch_size
    return noise_scale.item()


def compute_gradient_coherence(grads: List[torch.Tensor]) -> float:
    """
    Compute the cosine similarity (coherence) between consecutive gradients
    or a set of gradients to measure flow consistency.

    Args:
        grads: A list of flattened gradient tensors from consecutive steps.

    Returns:
        float: Average cosine similarity between consecutive gradients.
    """
    if len(grads) < 2:
        return 1.0

    similarities = []
    for i in range(len(grads) - 1):
        g1 = grads[i]
        g2 = grads[i+1]

        n1 = torch.norm(g1)
        n2 = torch.norm(g2)

        if n1 < 1e-10 or n2 < 1e-10:
            similarities.append(0.0)
        else:
            sim = torch.dot(g1, g2) / (n1 * n2)
            similarities.append(sim.item())

    return float(np.mean(similarities))


class GradientTracker:
    """
    Helper class to track gradients during the training loop.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.gradient_norms: List[float] = []

    def log_gradient_norm(self):
        """Extract and store the total gradient norm for the current step."""
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().data.norm(2).item() ** 2
        self.gradient_norms.append(total_norm_sq ** 0.5)

    def get_norms(self) -> List[float]:
        return self.gradient_norms
