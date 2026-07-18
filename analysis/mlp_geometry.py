import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

def track_mlp_activations(model: nn.Module, dataloader, device, num_batches: int = 10) -> Dict[str, torch.Tensor]:
    """
    Track MLP neuron activation frequencies across inputs via forward hooks.
    Counts the frequency of non-zero outputs from ReLU/GELU activations.
    """
    activations = defaultdict(list)
    hooks = []

    # Define hook function
    def get_activation(name):
        def hook(model, input, output):
            # Record non-zero activations (assumes output after activation function)
            # Or if hooking linear layer, we apply activation
            if isinstance(output, torch.Tensor):
                # Apply gelu if we're hooking linear1 (pre-activation)
                if "linear1" in name or "fc1" in name:
                    act = torch.nn.functional.gelu(output)
                else:
                    act = output

                # Check which neurons are active (value > 1e-4 to avoid numerical noise)
                is_active = (act > 1e-4).float()
                # Average over batch and sequence dimensions
                if is_active.dim() > 2:
                    # (batch, seq, d_ff)
                    active_freq = is_active.mean(dim=(0, 1))
                else:
                    active_freq = is_active.mean(dim=0)

                activations[name].append(active_freq.cpu())
        return hook

    # Register hooks on MLP linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "transformer" in name and ("linear1" in name or "fc1" in name):
            hooks.append(module.register_forward_hook(get_activation(name)))

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Aggregate results
    result = {}
    for name, acts in activations.items():
        if acts:
            result[name] = torch.stack(acts).mean(dim=0)

    return result

def compute_polysemanticity(model: nn.Module, dataloader, device, num_features: int) -> Dict[str, torch.Tensor]:
    """
    Estimate polysemanticity by computing the correlation between neuron activations
    and different output features (targets or logits).
    """
    activations = defaultdict(list)
    targets_list = []
    hooks = []

    def get_activation(name):
        def hook(model, input, output):
            if "linear1" in name or "fc1" in name:
                act = torch.nn.functional.gelu(output)
                # Max pool over sequence length to get 1 value per sequence
                if act.dim() > 2:
                    act = act.max(dim=1)[0]
                activations[name].append(act.cpu())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "transformer" in name and ("linear1" in name or "fc1" in name):
            hooks.append(module.register_forward_hook(get_activation(name)))

    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            targets_list.append(targets.cpu())

    for hook in hooks:
        hook.remove()

    targets_tensor = torch.cat(targets_list)

    # One-hot encode targets to use as "features"
    features = torch.nn.functional.one_hot(targets_tensor, num_classes=num_features).float()

    poly_scores = {}
    for name, acts in activations.items():
        acts_tensor = torch.cat(acts)  # (N, d_ff)

        # Center the data
        acts_centered = acts_tensor - acts_tensor.mean(dim=0, keepdim=True)
        feats_centered = features - features.mean(dim=0, keepdim=True)

        # Compute correlations
        acts_norm = torch.norm(acts_centered, dim=0, keepdim=True) + 1e-8
        feats_norm = torch.norm(feats_centered, dim=0, keepdim=True) + 1e-8

        acts_centered = acts_centered / acts_norm
        feats_centered = feats_centered / feats_norm

        # Correlation matrix: (d_ff, N) x (N, num_features) -> (d_ff, num_features)
        corr = torch.matmul(acts_centered.t(), feats_centered)

        # Polysemanticity score: number of features a neuron strongly correlates with (abs(corr) > 0.3)
        # Or just the entropy of the absolute correlations
        abs_corr = torch.abs(corr)
        # Normalize to probability distribution for entropy
        norm_corr = abs_corr / (abs_corr.sum(dim=1, keepdim=True) + 1e-8)

        # Entropy
        entropy = -(norm_corr * torch.log(norm_corr + 1e-8)).sum(dim=1)
        poly_scores[name] = entropy

    return poly_scores

def compute_weight_distance_from_init(model: nn.Module, init_model: nn.Module) -> Dict[str, float]:
    """
    Compute L2 distance between current weights and initialization.
    """
    distances = {}

    init_state = init_model.state_dict()
    curr_state = model.state_dict()

    for name, param in curr_state.items():
        if name in init_state and 'weight' in name:
            init_param = init_state[name]
            dist = torch.norm(param.float() - init_param.float()).item()
            distances[name] = dist

    # Aggregate by component
    agg_dist = {
        "embedding": 0.0,
        "attention": 0.0,
        "mlp": 0.0,
        "output_head": 0.0
    }

    for name, dist in distances.items():
        if "embed" in name:
            agg_dist["embedding"] += dist**2
        elif "transformer" in name and ("self_attn" in name or "attention" in name or "in_proj" in name or "out_proj" in name):
            agg_dist["attention"] += dist**2
        elif "transformer" in name and "linear" in name:
            agg_dist["mlp"] += dist**2
        elif "output_head" in name:
            agg_dist["output_head"] += dist**2

    for k in agg_dist:
        agg_dist[k] = agg_dist[k]**0.5

    return agg_dist
