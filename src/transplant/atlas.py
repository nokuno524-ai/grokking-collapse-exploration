"""
Lesion/ablation follow-up functions for causal behavior testing.
"""
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.transplant.patching import keys_for, COMPONENT_PATTERNS


def zero_ablate_component(model: nn.Module, component: str) -> None:
    """
    Zero out the weights and biases of a specific component in the model.
    """
    sd = model.state_dict()
    for k in keys_for(component, sd):
        sd[k].zero_()
    model.load_state_dict(sd)


def mean_ablate_component(model: nn.Module, component: str, dataloader: DataLoader, device: torch.device) -> None:
    """
    Mean-ablate a specific component in the model.
    Instead of zeroing, we set the weights to their mean value across the dataset (or just the parameter mean as a proxy for this simple setup).
    """
    sd = model.state_dict()
    for k in keys_for(component, sd):
        mean_val = sd[k].mean()
        sd[k].fill_(mean_val)
    model.load_state_dict(sd)


def resample_ablate_component(model: nn.Module, component: str, other_sd: Dict[str, torch.Tensor]) -> None:
    """
    Resample-ablate a component.
    Replaces the grafted component with the component from `other_sd` (typically the original recipient's un-grafted state or a random model).
    """
    from src.transplant.patching import patch_state_dict
    sd = model.state_dict()
    new_sd = patch_state_dict(sd, other_sd, component)
    model.load_state_dict(new_sd)


def evaluate_ablations(
    model: nn.Module,
    donor_sd: Dict[str, torch.Tensor],
    recipient_sd: Dict[str, torch.Tensor],
    components: List[str],
    eval_fn,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Test whether transplanted behavior causally depends on the component.

    1. First we graft the component from donor_sd.
    2. Then we record the behavior.
    3. Then we ablate the component (zero, mean, resample) and see if the behavior disappears.
    """
    results = {}

    # Needs to be implemented with graft_context
    from src.transplant.sweep import graft_context

    for comp in components:
        comp_results = {}

        # 1. Graft and eval
        with graft_context(model, donor_sd, [comp]):
            comp_results["grafted"] = eval_fn(model)

            # 2. Zero ablate the grafted model
            original_grafted_sd = {k: v.clone() for k, v in model.state_dict().items()}
            try:
                zero_ablate_component(model, comp)
                comp_results["zero_ablated"] = eval_fn(model)
            finally:
                model.load_state_dict(original_grafted_sd)

            # 3. Mean ablate the grafted model
            try:
                mean_ablate_component(model, comp, dataloader, device)
                comp_results["mean_ablated"] = eval_fn(model)
            finally:
                model.load_state_dict(original_grafted_sd)

            # 4. Resample ablate the grafted model (using original recipient weights)
            try:
                resample_ablate_component(model, comp, recipient_sd)
                comp_results["resample_ablated"] = eval_fn(model)
            finally:
                model.load_state_dict(original_grafted_sd)

        results[comp] = comp_results

    return results
