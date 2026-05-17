import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

def get_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Calculate the L2 norm of the gradients for each named parameter in the model.
    Also computes the 'total' L2 gradient norm.

    Args:
        model: PyTorch model after loss.backward() has been called.

    Returns:
        Dictionary mapping parameter names to their gradient L2 norm, including 'total'.
    """
    norms = {}
    total_squared_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2).item()
            norms[name] = norm
            total_squared_norm += norm ** 2
    norms['total'] = total_squared_norm ** 0.5
    return norms

def estimate_gradient_noise_scale(batch_gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """
    Estimate the gradient noise scale given a list of per-batch gradients.
    Approximated as the variance of the gradients across batches.

    Args:
        batch_gradients: List of dictionaries mapping parameter names to their gradient tensors.
                         Each dictionary corresponds to a different mini-batch.

    Returns:
        Dictionary mapping parameter names to their estimated gradient noise scale (variance).
    """
    if not batch_gradients:
        return {}

    param_names = batch_gradients[0].keys()
    noise_scales = {}

    for name in param_names:
        # Stack gradients for this parameter across all batches
        # Check if the gradient exists for all batches
        if all(name in batch for batch in batch_gradients):
            grads = torch.stack([batch[name] for batch in batch_gradients])
            # Variance across the batch dimension (dim=0)
            # We take the mean of the variances of each element to get a scalar noise scale
            variance = torch.var(grads, dim=0, unbiased=True).mean().item()
            noise_scales[name] = variance

    return noise_scales

def calculate_gradient_coherence(grad_t1: Dict[str, torch.Tensor], grad_t2: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Calculate the cosine similarity between gradients at two consecutive steps.
    Measures how consistently the optimization is moving in the same direction.

    Args:
        grad_t1: Dictionary of gradients at step t.
        grad_t2: Dictionary of gradients at step t+1.

    Returns:
        Dictionary mapping parameter names to their gradient cosine similarity, including 'total'.
    """
    coherence = {}

    dot_product_total = 0.0
    norm_t1_total = 0.0
    norm_t2_total = 0.0

    for name in grad_t1.keys():
        if name in grad_t2:
            g1 = grad_t1[name].view(-1)
            g2 = grad_t2[name].view(-1)

            dot = torch.dot(g1, g2).item()
            n1 = torch.norm(g1).item()
            n2 = torch.norm(g2).item()

            if n1 > 0 and n2 > 0:
                coherence[name] = dot / (n1 * n2)
            else:
                coherence[name] = 0.0

            dot_product_total += dot
            norm_t1_total += n1 ** 2
            norm_t2_total += n2 ** 2

    if norm_t1_total > 0 and norm_t2_total > 0:
        coherence['total'] = dot_product_total / (np.sqrt(norm_t1_total) * np.sqrt(norm_t2_total))
    else:
        coherence['total'] = 0.0

    return coherence

def detect_gradient_vanishing_explosion(model: nn.Module, vanishing_threshold: float = 1e-6, exploding_threshold: float = 1e3) -> Dict[str, str]:
    """
    Detect if gradients are vanishing or exploding for any parameter.

    Args:
        model: PyTorch model after backward pass.
        vanishing_threshold: Threshold below which gradients are considered vanishing.
        exploding_threshold: Threshold above which gradients are considered exploding.

    Returns:
        Dictionary mapping parameter names to 'vanishing', 'exploding', or 'normal'.
    """
    status = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2).item()
            if norm < vanishing_threshold:
                status[name] = 'vanishing'
            elif norm > exploding_threshold:
                status[name] = 'exploding'
            else:
                status[name] = 'normal'
    return status
