import torch
import torch.nn as nn
from typing import Dict, List

def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """Computes the gradient norm for each parameter in the model."""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
        else:
            grad_norms[name] = 0.0
    return grad_norms
