import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union

def center_kernel(K: np.ndarray) -> np.ndarray:
    """
    Centers a kernel matrix.

    Args:
        K: Kernel matrix of shape (n, n)

    Returns:
        Centered kernel matrix.
    """
    n = K.shape[0]
    # Centering matrix H = I - 1/n * 11^T
    H = np.eye(n) - np.ones((n, n)) / n
    # Centered K = H * K * H
    return H @ K @ H

def compute_linear_cka(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes Linear Centered Kernel Alignment (CKA) between two representation matrices.

    Args:
        X: Representations from model A, shape (num_samples, num_features_A)
        Y: Representations from model B, shape (num_samples, num_features_B)

    Returns:
        Linear CKA score between 0 and 1.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    # Ensure they have the same number of samples
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Number of samples must match. Got {X.shape[0]} and {Y.shape[0]}")

    # Center the representations
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    Y_centered = Y - np.mean(Y, axis=0, keepdims=True)

    # Compute dot products
    dot_product_XY = np.linalg.norm(X_centered.T @ Y_centered, ord='fro') ** 2
    norm_X = np.linalg.norm(X_centered.T @ X_centered, ord='fro')
    norm_Y = np.linalg.norm(Y_centered.T @ Y_centered, ord='fro')

    cka = dot_product_XY / (norm_X * norm_Y + 1e-10)
    return float(cka)

def compute_representation_rank(
    X: Union[np.ndarray, torch.Tensor],
    variance_threshold: float = 0.99
) -> Tuple[int, float]:
    """
    Measures the representation rank via Singular Value Decomposition.

    Args:
        X: Representation matrix of shape (num_samples, num_features).
        variance_threshold: The fraction of variance to explain.

    Returns:
        Tuple of (effective_rank, shannon_entropy).
        effective_rank: Number of singular values needed to explain `variance_threshold` of variance.
        shannon_entropy: Shannon entropy of normalized singular values.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # Center representations
    X_centered = X - np.mean(X, axis=0, keepdims=True)

    # SVD
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)

    # Calculate variance explained
    eigenvalues = s ** 2
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / (total_variance + 1e-10)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Effective rank based on threshold
    eff_rank = int(np.argmax(cumulative_variance >= variance_threshold) + 1)

    # Shannon entropy of normalized singular values
    s_norm = s / (np.sum(s) + 1e-10)
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))

    return eff_rank, float(np.exp(entropy))

def track_geometry_changes(
    representations_over_time: List[np.ndarray],
    variance_threshold: float = 0.99
) -> Dict[str, List[float]]:
    """
    Tracks how the geometry of the representation space changes over time
    (e.g., to observe collapse).

    Args:
        representations_over_time: List of representation matrices from consecutive steps.
        variance_threshold: Threshold for effective rank computation.

    Returns:
        Dict containing 'cka_to_previous', 'effective_rank', 'rank_entropy'.
    """
    results = {
        "cka_to_previous": [1.0],  # First step is perfectly aligned with itself
        "effective_rank": [],
        "rank_entropy": []
    }

    # Process first representation
    if representations_over_time:
        rank, entropy = compute_representation_rank(representations_over_time[0], variance_threshold)
        results["effective_rank"].append(rank)
        results["rank_entropy"].append(entropy)

    # Process remaining
    for i in range(1, len(representations_over_time)):
        prev_rep = representations_over_time[i-1]
        curr_rep = representations_over_time[i]

        cka = compute_linear_cka(prev_rep, curr_rep)
        rank, entropy = compute_representation_rank(curr_rep, variance_threshold)

        results["cka_to_previous"].append(cka)
        results["effective_rank"].append(rank)
        results["rank_entropy"].append(entropy)

    return results

def plot_cka_matrix(
    cka_matrix: np.ndarray,
    step_labels: List[Union[int, str]],
    save_path: str,
    title: str = "CKA Similarity Between Checkpoints"
):
    """
    Generates a heatmap of CKA similarity between different representations.

    Args:
        cka_matrix: Square matrix of CKA scores.
        step_labels: Labels for the ticks (e.g., training steps).
        save_path: File path to save the plot.
        title: Title of the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cka_matrix, cmap='magma', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Linear CKA')

    # Ticks
    if len(step_labels) <= 20:
        plt.xticks(np.arange(len(step_labels)), step_labels, rotation=45)
        plt.yticks(np.arange(len(step_labels)), step_labels)
    else:
        # Show fewer ticks if there are many steps
        idx = np.linspace(0, len(step_labels)-1, 10, dtype=int)
        plt.xticks(idx, [step_labels[i] for i in idx], rotation=45)
        plt.yticks(idx, [step_labels[i] for i in idx])

    plt.xlabel('Checkpoint Step')
    plt.ylabel('Checkpoint Step')
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_all_pairs_cka(representations: List[np.ndarray]) -> np.ndarray:
    """
    Computes pairwise CKA between all representations in the list.

    Args:
        representations: List of representation matrices.

    Returns:
        Square numpy array of CKA similarities.
    """
    n = len(representations)
    cka_matrix = np.zeros((n, n))

    for i in range(n):
        cka_matrix[i, i] = 1.0  # CKA with self is always 1
        for j in range(i + 1, n):
            cka = compute_linear_cka(representations[i], representations[j])
            cka_matrix[i, j] = cka
            cka_matrix[j, i] = cka

    return cka_matrix
