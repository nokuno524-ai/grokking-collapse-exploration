import numpy as np
from typing import List, Dict, Any
import torch

def compute_hessian_trace_proxy(gradients: List[np.ndarray]) -> float:
    """
    Approximate landscape sharpness (Hessian trace) using variance of gradients.
    """
    if not gradients or len(gradients) == 0:
        return 0.0

    grad_array = np.array(gradients) # shape (steps, params)
    variances = np.var(grad_array, axis=0)
    trace_proxy = np.sum(variances)

    return float(trace_proxy)

def compute_effective_learning_rate(model_state_prev: Dict[str, Any], model_state_curr: Dict[str, Any], lr: float, gradients: Dict[str, Any] = None) -> float:
    """
    Compute effective learning rate (actual step size in function space or parameter space).
    If gradients are provided, computes ||W_t+1 - W_t|| / ||g_t||.
    If not, just returns the L2 norm of the difference.
    """
    diff_sq_sum = 0.0
    for k in model_state_curr:
        if k in model_state_prev and isinstance(model_state_curr[k], torch.Tensor):
            diff = model_state_curr[k].float() - model_state_prev[k].float()
            diff_sq_sum += torch.sum(diff**2).item()

    dist = np.sqrt(diff_sq_sum)

    if gradients:
        grad_sq_sum = 0.0
        for k, g in gradients.items():
            if isinstance(g, torch.Tensor):
                grad_sq_sum += torch.sum(g**2).item()
        grad_norm = np.sqrt(grad_sq_sum)
        if grad_norm > 1e-10:
            return float(dist / grad_norm)

    return float(dist)

def measure_weight_velocity(history_states: List[Dict[str, Any]]) -> List[float]:
    """
    Compute L2 norm of parameter differences between steps (velocity).
    """
    velocities = []

    for i in range(1, len(history_states)):
        prev = history_states[i-1]
        curr = history_states[i]

        diff_sq_sum = 0.0
        for k in curr:
            if k in prev and isinstance(curr[k], torch.Tensor):
                diff = curr[k].float() - prev[k].float()
                diff_sq_sum += torch.sum(diff**2).item()

        velocities.append(float(np.sqrt(diff_sq_sum)))

    return velocities
