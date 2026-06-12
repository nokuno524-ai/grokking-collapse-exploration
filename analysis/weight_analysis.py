import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple

def compute_weight_norms(weight: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    Computes L1, L2 (spectral), and Frobenius norms of a weight matrix.

    Args:
        weight: The weight array or tensor.

    Returns:
        Dictionary containing 'l1', 'l2', and 'frobenius' norms.
    """
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()

    l1_norm = float(np.sum(np.abs(weight)))

    if weight.ndim >= 2:
        fro_norm = float(np.linalg.norm(weight, ord='fro'))
        # Spectral norm
        s = np.linalg.svd(weight, compute_uv=False)
        l2_norm = float(np.max(s))
    else:
        fro_norm = float(np.linalg.norm(weight))
        l2_norm = fro_norm

    return {
        "l1": l1_norm,
        "l2": l2_norm,
        "frobenius": fro_norm
    }

def compute_effective_rank(weight: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Computes the effective rank of a weight matrix using the Shannon entropy
    of its normalized singular values.

    Args:
        weight: The weight array or tensor.

    Returns:
        The effective rank (float).
    """
    if isinstance(weight, torch.Tensor):
        weight = weight.detach().cpu().numpy()

    if weight.ndim < 2:
        return 1.0

    # Standardize weights
    s = np.linalg.svd(weight, compute_uv=False)
    s_norm = s / (np.sum(s) + 1e-10)
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
    return float(np.exp(entropy))

def load_weights(checkpoint_path: str) -> Dict[str, np.ndarray]:
    """
    Loads model weights from a PyTorch checkpoint (.pt, .pth) or numpy archive (.npz).

    Args:
        checkpoint_path: Path to the weights file.

    Returns:
        Dictionary mapping layer names to numpy arrays of weights.
    """
    if checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth'):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model_state', ckpt)
        if 'model_state_dict' in ckpt:
             state_dict = ckpt['model_state_dict']
        return {k: v.cpu().numpy() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    elif checkpoint_path.endswith('.npz'):
        with np.load(checkpoint_path) as data:
            return {k: v for k, v in data.items()}
    else:
        raise ValueError("Unsupported file format. Use .pt, .pth, or .npz")

def analyze_checkpoints(checkpoints: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Computes norms and effective rank for all layers across a list of checkpoints.

    Args:
        checkpoints: List of paths to checkpoint files.

    Returns:
        Nested dictionary tracking metrics over time per layer.
    """
    results = {}
    for path in checkpoints:
        weights = load_weights(path)
        for layer_name, w in weights.items():
            if layer_name not in results:
                results[layer_name] = {"l1": [], "l2": [], "frobenius": [], "effective_rank": []}

            norms = compute_weight_norms(w)
            rank = compute_effective_rank(w)

            results[layer_name]["l1"].append(norms["l1"])
            results[layer_name]["l2"].append(norms["l2"])
            results[layer_name]["frobenius"].append(norms["frobenius"])
            results[layer_name]["effective_rank"].append(rank)

    return results

def compare_models(model_a_weights: Dict[str, np.ndarray], model_b_weights: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Tracks weight norm ratio between two models (e.g., collapsed vs non-collapsed).

    Args:
        model_a_weights: Dictionary of weights for Model A.
        model_b_weights: Dictionary of weights for Model B.

    Returns:
        Dictionary mapping layer names to their Frobenius norm ratio (A / B).
    """
    ratios = {}
    for layer in model_a_weights:
        if layer in model_b_weights:
            norm_a = compute_weight_norms(model_a_weights[layer])["frobenius"]
            norm_b = compute_weight_norms(model_b_weights[layer])["frobenius"]
            ratios[layer] = norm_a / (norm_b + 1e-10)
    return ratios

def identify_divergent_layers(ratios: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Identifies layers with the largest norm divergence (ratio furthest from 1.0).

    Args:
        ratios: Dictionary mapping layer names to their norm ratio.
        top_k: Number of top divergent layers to return.

    Returns:
        List of tuples (layer_name, ratio) sorted by divergence.
    """
    divergences = {layer: abs(ratio - 1.0) for layer, ratio in ratios.items()}
    sorted_layers = sorted(divergences.items(), key=lambda x: x[1], reverse=True)
    top_layers = [layer for layer, _ in sorted_layers[:top_k]]
    return [(layer, ratios[layer]) for layer in top_layers]

def plot_weight_evolution(results: Dict[str, Dict[str, List[float]]], metric: str, save_path: str):
    """
    Generates weight norm evolution plots per layer and saves to a file.

    Args:
        results: Results dictionary from analyze_checkpoints.
        metric: Metric to plot ('l1', 'l2', 'frobenius', or 'effective_rank').
        save_path: File path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    for layer_name, metrics in results.items():
        if metric in metrics:
            plt.plot(metrics[metric], label=layer_name)

    plt.title(f"Weight Evolution: {metric}")
    plt.xlabel("Checkpoint Step")
    plt.ylabel(metric)

    if len(results) <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
