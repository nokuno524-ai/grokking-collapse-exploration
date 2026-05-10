import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Dict, List, Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def compute_weight_norms(model: nn.Module) -> Dict[str, float]:
    """Compute L1 and L2 norms per layer."""
    norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            l1_norm = param.norm(p=1).item()
            l2_norm = param.norm(p=2).item()
            norms[f"{name}_l1"] = l1_norm
            norms[f"{name}_l2"] = l2_norm
    return norms

def compute_weight_rank(model: nn.Module, threshold: float = 0.99) -> Dict[str, int]:
    """Compute the effective rank of 2D weight matrices."""
    ranks = {}
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) == 2:
            W = param.detach().cpu()
            # Perform SVD
            try:
                s = torch.linalg.svdvals(W)
                # Compute effective rank based on energy threshold
                s_sq = s ** 2
                energy = s_sq.cumsum(dim=0) / s_sq.sum()
                rank = (energy >= threshold).nonzero()[0].item() + 1
                ranks[f"{name}_rank"] = int(rank)
            except RuntimeError:
                pass # SVD did not converge
    return ranks

def compute_condition_number(model: nn.Module) -> Dict[str, float]:
    """Compute the condition number of 2D weight matrices."""
    condition_nums = {}
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) == 2:
            W = param.detach().cpu()
            try:
                s = torch.linalg.svdvals(W)
                if s[-1] > 0:
                    condition_nums[f"{name}_cond"] = (s[0] / s[-1]).item()
                else:
                    condition_nums[f"{name}_cond"] = float('inf')
            except RuntimeError:
                pass
    return condition_nums

def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """Compute L2 norms of gradients per layer."""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norms[f"{name}_grad_l2"] = param.grad.norm(p=2).item()
        elif param.requires_grad:
            grad_norms[f"{name}_grad_l2"] = 0.0
    return grad_norms

def track_weight_evolution(model_snapshots: List[nn.Module]) -> Dict[str, Any]:
    """Track how weights change over training snapshots."""
    evolution = {"norms": [], "ranks": [], "condition_numbers": []}
    for model in model_snapshots:
        evolution["norms"].append(compute_weight_norms(model))
        evolution["ranks"].append(compute_weight_rank(model))
        evolution["condition_numbers"].append(compute_condition_number(model))
    return evolution

def detect_collapse_from_weights(weight_history: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
    """
    Detect collapse signatures such as rapidly decreasing effective rank
    or exploding condition numbers.
    """
    signatures = {"collapse_detected": False, "reason": None}

    if not weight_history.get("ranks") or len(weight_history["ranks"]) < 2:
        return signatures

    # Analyze rank dropping
    first_ranks = weight_history["ranks"][0]
    last_ranks = weight_history["ranks"][-1]

    for layer_name in first_ranks:
        if layer_name in last_ranks:
            start_rank = first_ranks[layer_name]
            end_rank = last_ranks[layer_name]
            # If rank drops by more than 50%, flag as collapse signature
            if start_rank > 0 and end_rank / start_rank < 0.5:
                signatures["collapse_detected"] = True
                signatures["reason"] = f"Significant rank drop in {layer_name}: {start_rank} -> {end_rank}"
                return signatures

    # Analyze condition number exploding
    if weight_history.get("condition_numbers") and len(weight_history["condition_numbers"]) >= 2:
        first_cond = weight_history["condition_numbers"][0]
        last_cond = weight_history["condition_numbers"][-1]

        for layer_name in first_cond:
            if layer_name in last_cond:
                start_c = first_cond[layer_name]
                end_c = last_cond[layer_name]
                # If condition number explodes, flag as potential collapse
                if start_c > 0 and end_c > start_c * 10:
                    signatures["collapse_detected"] = True
                    signatures["reason"] = f"Condition number explosion in {layer_name}"
                    return signatures

    return signatures

def plot_weight_analysis(analysis: Dict[str, Any], output_dir: str):
    """Multi-panel text-based or minimal plotting for weight analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Save raw analysis to JSON
    with open(os.path.join(output_dir, "weight_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    if not HAS_MATPLOTLIB:
        print(f"Text-based plot summary saved to {output_dir}/weight_analysis.json")
        return

    # Simple multi-panel plot
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot norms evolution
        if analysis.get("norms"):
            norms_data = analysis["norms"]
            steps = range(len(norms_data))
            for key in norms_data[0].keys():
                if "_l2" in key:
                    vals = [n.get(key, 0) for n in norms_data]
                    axes[0].plot(steps, vals, label=key)
            axes[0].set_title("L2 Norms Evolution")
            axes[0].legend(fontsize='small', loc='best')

        # Plot rank evolution
        if analysis.get("ranks"):
            ranks_data = analysis["ranks"]
            steps = range(len(ranks_data))
            for key in ranks_data[0].keys():
                vals = [r.get(key, 0) for r in ranks_data]
                axes[1].plot(steps, vals, label=key)
            axes[1].set_title("Effective Rank Evolution")
            axes[1].legend(fontsize='small', loc='best')

        # Plot condition numbers
        if analysis.get("condition_numbers"):
            cond_data = analysis["condition_numbers"]
            steps = range(len(cond_data))
            for key in cond_data[0].keys():
                vals = [c.get(key, 0) for c in cond_data]
                # Filter out infs and <=0 for log plotting
                vals = [v if v != float('inf') and v > 0 else np.nan for v in vals]
                axes[2].plot(steps, vals, label=key)
            axes[2].set_title("Condition Number Evolution")
            axes[2].set_yscale('log')
            axes[2].legend(fontsize='small', loc='best')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "weight_analysis_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Failed to generate plot: {e}")
