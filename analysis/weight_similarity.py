import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Union
import os


def compute_cka(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> float:
    """
    Computes Linear Centered Kernel Alignment (CKA) between two matrices.
    Requires matrices to be 2D (num_examples x num_features).

    Args:
        matrix_a: First tensor.
        matrix_b: Second tensor.

    Returns:
        CKA similarity score (between 0 and 1).
    """
    # Flatten if not 2D
    if matrix_a.dim() != 2:
        matrix_a = matrix_a.view(matrix_a.size(0), -1)
    if matrix_b.dim() != 2:
        matrix_b = matrix_b.view(matrix_b.size(0), -1)

    # Ensure they have same number of examples
    assert matrix_a.size(0) == matrix_b.size(0), "Matrices must have same number of rows/examples"

    # Center the matrices (subtract column means)
    matrix_a = matrix_a - matrix_a.mean(dim=0, keepdim=True)
    matrix_b = matrix_b - matrix_b.mean(dim=0, keepdim=True)

    # Compute dot products (Gram matrices)
    # Using Frobenius norm formulation which is equivalent and often faster
    # <X^T Y, X^T Y>_F / (||X^T X||_F * ||Y^T Y||_F)

    # Calculate cross term
    dot_product = torch.mm(matrix_a.t(), matrix_b)
    cross_norm = torch.norm(dot_product, p='fro') ** 2

    # Calculate auto terms
    norm_a = torch.norm(torch.mm(matrix_a.t(), matrix_a), p='fro')
    norm_b = torch.norm(torch.mm(matrix_b.t(), matrix_b), p='fro')

    if norm_a.item() == 0 or norm_b.item() == 0:
        return 0.0

    cka = cross_norm / (norm_a * norm_b)

    return float(cka.item())


def compute_weight_cka(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Computes CKA between corresponding weight matrices of two model states.
    For weights, we transpose so that "examples" is the output dimension and "features" is input dim.
    """
    similarities = {}

    for k in state_a.keys():
        if 'weight' in k and state_a[k].dim() >= 2:
            # For weights (out_features, in_features), treat out_features as 'examples'
            # so we compare the feature representations learned.
            w_a = state_a[k]
            w_b = state_b[k]

            # Reshape e.g. convs to (out_channels, -1)
            w_a_flat = w_a.view(w_a.size(0), -1)
            w_b_flat = w_b.view(w_b.size(0), -1)

            try:
                sim = compute_cka(w_a_flat, w_b_flat)
                similarities[k] = sim
            except Exception as e:
                print(f"Warning: Failed to compute CKA for {k}: {e}")

    return similarities


def track_similarity_trajectory(checkpoints: List[Dict[str, Any]],
                                reference_state: Dict[str, torch.Tensor] = None) -> Dict[str, List[float]]:
    """
    Tracks the CKA similarity of weights over training.
    If reference_state is None, compares each checkpoint to the initial checkpoint (index 0).
    Otherwise, compares each checkpoint to the reference_state (e.g. final grokked model).
    """
    if not checkpoints:
        return {}

    trajectories = {}

    if reference_state is None:
        reference_state = checkpoints[0].get('model_state', checkpoints[0])

    for ckpt in checkpoints:
        state = ckpt.get('model_state', ckpt)
        sims = compute_weight_cka(state, reference_state)

        for k, sim in sims.items():
            if k not in trajectories:
                trajectories[k] = []
            trajectories[k].append(sim)

    return trajectories


def compare_cka_trajectories(grokked_trajectories: Dict[str, List[float]],
                             collapsed_trajectories: Dict[str, List[float]],
                             save_path: str = None):
    """
    Plots CKA trajectories side-by-side.
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
        ax.set_ylabel('CKA Similarity')
        ax.legend()
        ax.set_ylim(0, 1.05)

    for i in range(n_layers, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
