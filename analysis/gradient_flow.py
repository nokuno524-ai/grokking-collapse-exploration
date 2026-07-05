import torch
import torch.nn as nn
from typing import Dict, List, Optional

def compute_gradient_flow_approx(model_t1: nn.Module, model_t2: nn.Module, lr: float) -> Dict[str, float]:
    """
    Approximate gradient norms per layer using W_t - W_{t-1} when true gradients are unavailable.
    Assuming simple SGD-like updates where W_t - W_{t-1} ~ -lr * grad.
    Thus, grad ~ (W_{t-1} - W_t) / lr.
    """
    grad_norms = {}
    for (name1, p1), (name2, p2) in zip(model_t1.named_parameters(), model_t2.named_parameters()):
        assert name1 == name2
        if p1.requires_grad:
            approx_grad = (p1.detach() - p2.detach()) / lr
            grad_norms[name1] = approx_grad.norm().item()
    return grad_norms

def track_gradient_flow(checkpoints: List[str], model_class, model_kwargs: dict, lr: float) -> Dict[str, List[float]]:
    """
    Track gradient norm flow across multiple checkpoints.

    Args:
        checkpoints: Sorted list of checkpoint paths.
        model_class: Model class.
        model_kwargs: Model initialization arguments.
        lr: Learning rate used during training.

    Returns:
        Dict mapping parameter names to list of gradient norms.
    """
    if len(checkpoints) < 2:
        return {}

    flow_history = {}

    # Load first model
    model_prev = model_class(**model_kwargs)
    try:
        ckpt_prev = torch.load(checkpoints[0], weights_only=True)
    except:
        ckpt_prev = torch.load(checkpoints[0], weights_only=False)
    model_prev.load_state_dict(ckpt_prev["model_state"])

    for path in checkpoints[1:]:
        model_curr = model_class(**model_kwargs)
        try:
            ckpt_curr = torch.load(path, weights_only=True)
        except:
            ckpt_curr = torch.load(path, weights_only=False)
        model_curr.load_state_dict(ckpt_curr["model_state"])

        grad_norms = compute_gradient_flow_approx(model_prev, model_curr, lr)

        for name, norm in grad_norms.items():
            if name not in flow_history:
                flow_history[name] = []
            flow_history[name].append(norm)

        model_prev = model_curr

    return flow_history

if __name__ == "__main__":
    from src.model import ModularArithmeticTransformer
    m1 = ModularArithmeticTransformer()
    m2 = ModularArithmeticTransformer()
    # Randomly slightly modify m2
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    norms = compute_gradient_flow_approx(m1, m2, lr=1e-3)
    print(f"Computed gradient norms for {len(norms)} parameters.")
