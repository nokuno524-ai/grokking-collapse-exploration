import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from scipy import stats
import os


def compute_layer_norms(model_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Computes the L2 norm of each weight matrix in the model state.

    Args:
        model_state: Dictionary mapping parameter names to tensors (e.g., from checkpoint['model_state'])

    Returns:
        Dictionary mapping layer names to their L2 norm.
    """
    norms = {}
    for name, param in model_state.items():
        if 'weight' in name:
            norms[name] = float(torch.norm(param, p=2).item())
    return norms


def track_norm_trajectories(checkpoints: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Tracks the L2 norm of each layer's weights over a sequence of checkpoints.

    Args:
        checkpoints: List of checkpoint dictionaries, ordered by step.

    Returns:
        Dictionary mapping layer names to lists of L2 norms over time.
    """
    trajectories = {}

    for ckpt in checkpoints:
        state = ckpt.get('model_state', ckpt) # fallback if directly passing model state
        norms = compute_layer_norms(state)

        for name, norm in norms.items():
            if name not in trajectories:
                trajectories[name] = []
            trajectories[name].append(norm)

    return trajectories


def track_norm_distributions(checkpoints: List[Dict[str, Any]], layer_name: str = None) -> List[np.ndarray]:
    """
    Tracks the distribution (histogram) of weight values over time.

    Args:
        checkpoints: List of checkpoint dictionaries.
        layer_name: Specific layer to track. If None, tracks all weights concatenated.

    Returns:
        List of weight arrays (as numpy arrays) for each checkpoint.
    """
    distributions = []

    for ckpt in checkpoints:
        state = ckpt.get('model_state', ckpt)

        if layer_name:
            if layer_name in state:
                distributions.append(state[layer_name].cpu().numpy().flatten())
        else:
            all_weights = []
            for name, param in state.items():
                if 'weight' in name:
                    all_weights.append(param.cpu().numpy().flatten())
            if all_weights:
                distributions.append(np.concatenate(all_weights))

    return distributions


def compare_decay_paths(grokked_trajectories: Dict[str, List[float]],
                        collapsed_trajectories: Dict[str, List[float]],
                        save_path: str = None):
    """
    Plots and compares the weight decay paths for grokking vs collapse.

    Args:
        grokked_trajectories: Trajectories from a grokked model.
        collapsed_trajectories: Trajectories from a collapsed model.
        save_path: Optional path to save the figure.
    """
    layers = list(grokked_trajectories.keys())
    n_layers = len(layers)

    cols = min(3, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, layer in enumerate(layers):
        ax = axes[i]

        g_traj = grokked_trajectories[layer]
        c_traj = collapsed_trajectories.get(layer, [])

        ax.plot(g_traj, label='Grokked', color='blue', alpha=0.7)
        if c_traj:
            ax.plot(c_traj, label='Collapsed', color='red', alpha=0.7)

        ax.set_title(layer)
        ax.set_xlabel('Checkpoint Index')
        ax.set_ylabel('L2 Norm')
        ax.legend()

    for i in range(n_layers, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def check_norm_reduction_predictor(grokked_final_norms: List[float],
                                  collapsed_final_norms: List[float]) -> Tuple[float, float, str]:
    """
    Performs a statistical test (Mann-Whitney U) to determine if weight norm reduction
    is a reliable predictor of grokking failure.

    Args:
        grokked_final_norms: Final total weight norms of runs that successfully grokked.
        collapsed_final_norms: Final total weight norms of runs that collapsed.

    Returns:
        U-statistic, p-value, and a qualitative conclusion string.
    """
    # Using Mann-Whitney U test as distributions may not be normal
    stat, p_value = stats.mannwhitneyu(grokked_final_norms, collapsed_final_norms, alternative='two-sided')

    mean_g = np.mean(grokked_final_norms)
    mean_c = np.mean(collapsed_final_norms)

    if p_value < 0.05:
        if mean_c < mean_g:
            conclusion = f"Significant reduction: Collapsed models have significantly lower weight norms (p={p_value:.4f})."
        else:
            conclusion = f"Significant difference: Collapsed models have HIGHER weight norms (p={p_value:.4f})."
    else:
        conclusion = f"No significant difference in weight norms between grokked and collapsed models (p={p_value:.4f})."

    return stat, p_value, conclusion
