import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

def compute_mutual_information(activations: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes mutual information between activations and labels using histogram binning.

    Args:
        activations: A numpy array of shape (N, D) where N is number of samples and D is feature dim.
                     If 1D, shape should be (N,).
        labels: A numpy array of shape (N,)
        num_bins: Number of bins to discretize activations.

    Returns:
        Mutual information estimate in bits.
    """
    if len(activations.shape) == 1:
        activations = activations.reshape(-1, 1)

    N, D = activations.shape

    # We will compute mutual information for each dimension and average it
    # as a simple approximation for high-dimensional MI.
    # A more rigorous approach would require kernel density estimation or similar.
    total_mi = 0.0
    for d in range(D):
        # Discretize the d-th dimension
        act_d = activations[:, d]

        # Calculate joint histogram
        hist_2d, _, _ = np.histogram2d(act_d, labels, bins=(num_bins, len(np.unique(labels))))

        # Convert to probabilities
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        # Calculate MI
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
        total_mi += mi

    return total_mi / D

def compute_conditional_entropy(activations: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes the conditional entropy H(T|Y) of the activations given the labels.

    Args:
        activations: A numpy array of shape (N, D).
        labels: A numpy array of shape (N,).
        num_bins: Number of bins to discretize activations.

    Returns:
        Conditional entropy H(T|Y) in bits.
    """
    if len(activations.shape) == 1:
        activations = activations.reshape(-1, 1)

    N, D = activations.shape
    total_cond_entropy = 0.0

    for d in range(D):
        act_d = activations[:, d]
        hist_2d, _, _ = np.histogram2d(act_d, labels, bins=(num_bins, len(np.unique(labels))))

        pxy = hist_2d / float(np.sum(hist_2d))
        py = np.sum(pxy, axis=0)

        # H(T|Y) = -sum p(t,y) log(p(t|y)) = -sum p(t,y) log(p(t,y)/p(y))
        # where p(t|y) = p(t,y) / p(y)

        nzs = pxy > 0
        py_expanded = np.tile(py, (pxy.shape[0], 1))

        # p(t|y) calculation
        pty = pxy[nzs] / py_expanded[nzs]

        cond_entropy = -np.sum(pxy[nzs] * np.log2(pty))
        total_cond_entropy += cond_entropy

    return total_cond_entropy / D

def information_bottleneck_curve(
    activations_by_epoch: List[np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10
) -> Tuple[List[float], List[float]]:
    """
    Tracks the information bottleneck tradeoff (compression vs relevance) over epochs.

    Args:
        activations_by_epoch: List of activations (N, D) at different training epochs.
        inputs: Input data (N, ...)
        labels: Target labels (N,)
        num_bins: Number of bins for MI estimation.

    Returns:
        mi_xt_list: Mutual information between input and representation I(X; T) for each epoch.
        mi_ty_list: Mutual information between representation and label I(T; Y) for each epoch.
    """
    mi_xt_list = []
    mi_ty_list = []

    # Simple hack: treat input as label for I(X; T) calculation by combining its features if needed,
    # or compute MI per input dimension.
    # To keep it simple, we discretize inputs into 1D indices if they are integers
    # (which is typical for modular arithmetic inputs: (a, b)).
    if len(inputs.shape) > 1:
        # e.g. for mod arithmetic (N, 2), map to single integer
        # Assuming inputs are bounded by some max value P
        max_val = np.max(inputs) + 1
        inputs_1d = inputs[:, 0] * max_val + inputs[:, 1] if inputs.shape[1] == 2 else inputs[:, 0]
    else:
        inputs_1d = inputs

    for acts in activations_by_epoch:
        # Compression: I(X; T)
        mi_xt = compute_mutual_information(acts, inputs_1d, num_bins=num_bins)
        mi_xt_list.append(mi_xt)

        # Relevance: I(T; Y)
        mi_ty = compute_mutual_information(acts, labels, num_bins=num_bins)
        mi_ty_list.append(mi_ty)

    return mi_xt_list, mi_ty_list

def plot_information_flow(mi_xt_list: List[float], mi_ty_list: List[float], epochs: Optional[List[int]] = None, save_path: Optional[str] = None):
    """
    Plots the information bottleneck curve (I(X; T) vs I(T; Y)).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(mi_xt_list, mi_ty_list, 'o-', linewidth=2, markersize=6)

    # Add epoch annotations
    if epochs is None:
        epochs = list(range(len(mi_xt_list)))

    # Annotate every few points to avoid clutter
    step = max(1, len(epochs) // 10)
    for i in range(0, len(epochs), step):
        plt.annotate(f"E{epochs[i]}", (mi_xt_list[i], mi_ty_list[i]),
                     textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('I(X; T) - Compression (bits)')
    plt.ylabel('I(T; Y) - Relevance (bits)')
    plt.title('Information Bottleneck Tradeoff')
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Information flow plot saved to {save_path}")

    return plt.gcf()
