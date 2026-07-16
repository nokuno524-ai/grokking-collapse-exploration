import torch
import torch.nn as nn
from typing import Dict, List, Tuple

def approximate_gradients(model_prev: nn.Module, model_curr: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Approximates gradients using consecutive weight updates (W_t - W_{t-1}).

    Args:
        model_prev: Model at step t-1.
        model_curr: Model at step t.

    Returns:
        Dictionary mapping parameter names to their approximated gradients.
    """
    grads = {}
    for (name_prev, param_prev), (name_curr, param_curr) in zip(model_prev.named_parameters(), model_curr.named_parameters()):
        if name_prev != name_curr:
            raise ValueError("Model structures do not match.")

        # W_t - W_{t-1} is proportional to the negative gradient (assuming SGD for approximation)
        # We'll just return the difference as the "gradient step" direction
        grads[name_curr] = param_curr.data - param_prev.data

    return grads

def track_gradient_norms(checkpoints: List[Dict]) -> Dict[str, List[float]]:
    """
    Tracks gradient norms per layer over training using weight differences.

    Args:
        checkpoints: List of model state dicts.

    Returns:
        Dictionary mapping parameter names to lists of gradient norms over time.
    """
    from src.model import ModularArithmeticTransformer
    from collections import defaultdict

    if len(checkpoints) < 2:
        return {}

    norms_per_layer = defaultdict(list)

    model_prev = ModularArithmeticTransformer()
    model_curr = ModularArithmeticTransformer()

    for i in range(1, len(checkpoints)):
        model_prev.load_state_dict(checkpoints[i-1]['model_state'])
        model_curr.load_state_dict(checkpoints[i]['model_state'])

        grads = approximate_gradients(model_prev, model_curr)

        for name, grad in grads.items():
            norms_per_layer[name].append(torch.norm(grad, p=2).item())

    return dict(norms_per_layer)

def identify_gradient_starvation(grads_pure: Dict[str, torch.Tensor], grads_collapsed: Dict[str, torch.Tensor], threshold_ratio: float = 0.1) -> List[str]:
    """
    Identifies parameters suffering from gradient starvation in collapsed models.
    A parameter is starved if its gradient norm in the collapsed model is significantly
    smaller than in the pure model.

    Args:
        grads_pure: Approximated gradients for pure model.
        grads_collapsed: Approximated gradients for collapsed model.
        threshold_ratio: Ratio below which a gradient is considered starved.

    Returns:
        List of parameter names suffering from starvation.
    """
    starved_params = []

    for name in grads_pure:
        if name in grads_collapsed:
            norm_pure = torch.norm(grads_pure[name], p=2).item()
            norm_collapsed = torch.norm(grads_collapsed[name], p=2).item()

            # If norm_pure is 0, skip
            if norm_pure > 1e-10 and (norm_collapsed / norm_pure) < threshold_ratio:
                starved_params.append(name)

    return starved_params

def measure_gradient_noise_scale(model: nn.Module, data_loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device = torch.device("cpu")) -> float:
    """
    Measures gradient noise scale (McCandlish et al.) approximately.
    GNS = trace(Covariance(g)) / ||E[g]||^2

    Args:
        model: Model to evaluate.
        data_loader: DataLoader with batch size 1 to get per-example gradients.
        criterion: Loss function.
        device: Device to use.

    Returns:
        Gradient noise scale.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    if not params:
        return 0.0

    per_example_grads = []

    # Collect per-example gradients (warning: memory intensive for large datasets/models)
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        flat_grad = torch.cat([p.grad.contiguous().view(-1) for p in params if p.grad is not None])
        per_example_grads.append(flat_grad)

    if not per_example_grads:
        return 0.0

    grads_tensor = torch.stack(per_example_grads)  # (N, num_params)

    mean_grad = grads_tensor.mean(dim=0)
    mean_grad_norm_sq = torch.norm(mean_grad) ** 2

    if mean_grad_norm_sq.item() < 1e-10:
        return 0.0

    # Empirical trace of covariance
    # sum_{i=1}^N ||g_i - mean_g||^2 / (N-1)
    variance_term = torch.sum(torch.norm(grads_tensor - mean_grad, dim=1) ** 2) / max(1, len(per_example_grads) - 1)

    gns = variance_term / mean_grad_norm_sq
    return gns.item()

def plot_gradient_cosine_similarity(checkpoints: List[Dict], output_path: str = None) -> List[float]:
    """
    Computes and plots gradient cosine similarity between consecutive steps.

    Args:
        checkpoints: List of model state dicts.
        output_path: Optional path to save the plot.

    Returns:
        List of cosine similarities. Length will be len(checkpoints) - 2.
    """
    from src.model import ModularArithmeticTransformer
    import matplotlib.pyplot as plt

    if len(checkpoints) < 3:
        return []

    sims = []

    model_t0 = ModularArithmeticTransformer()
    model_t1 = ModularArithmeticTransformer()
    model_t2 = ModularArithmeticTransformer()

    for i in range(2, len(checkpoints)):
        model_t0.load_state_dict(checkpoints[i-2]['model_state'])
        model_t1.load_state_dict(checkpoints[i-1]['model_state'])
        model_t2.load_state_dict(checkpoints[i]['model_state'])

        grads_1 = approximate_gradients(model_t0, model_t1)
        grads_2 = approximate_gradients(model_t1, model_t2)

        flat_1 = torch.cat([g.contiguous().view(-1) for g in grads_1.values()])
        flat_2 = torch.cat([g.contiguous().view(-1) for g in grads_2.values()])

        if torch.norm(flat_1) == 0 or torch.norm(flat_2) == 0:
            sims.append(0.0)
        else:
            sim = torch.nn.functional.cosine_similarity(flat_1, flat_2, dim=0)
            sims.append(sim.item())

    if output_path:
        plt.figure(figsize=(10, 6))
        plt.plot(sims, marker='o')
        plt.title('Gradient Cosine Similarity Between Steps')
        plt.xlabel('Step Index')
        plt.ylabel('Cosine Similarity')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

    return sims
