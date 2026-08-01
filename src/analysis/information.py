import torch
import numpy as np

def center_gram(gram: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix."""
    n = gram.size(0)
    h = torch.eye(n, device=gram.device) - torch.ones((n, n), device=gram.device) / n
    return h @ gram @ h

def compute_cka(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """
    Compute Linear Centered Kernel Alignment (CKA) between two feature matrices.
    """
    # Assuming shape (batch_size, features)
    # If 3D, flatten the spatial/sequence dimensions
    if features_x.dim() > 2:
        features_x = features_x.view(features_x.size(0), -1)
    if features_y.dim() > 2:
        features_y = features_y.view(features_y.size(0), -1)

    gram_x = features_x @ features_x.T
    gram_y = features_y @ features_y.T

    gram_x_c = center_gram(gram_x)
    gram_y_c = center_gram(gram_y)

    # Compute dot product of vectorized matrices
    scaled_hsic = torch.sum(gram_x_c * gram_y_c)

    norm_x = torch.sqrt(torch.sum(gram_x_c * gram_x_c))
    norm_y = torch.sqrt(torch.sum(gram_y_c * gram_y_c))

    cka = scaled_hsic / (norm_x * norm_y + 1e-10)
    return cka.item()

def compute_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Compute mutual information between two arrays using histogram binning.
    """
    # Flatten arrays
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Calculate 2D histogram
    c_xy = np.histogram2d(x_flat, y_flat, bins)[0]

    # Convert to probabilities
    p_xy = c_xy / np.sum(c_xy)

    # Marginal probabilities
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    # Calculate mutual information
    # Only keep non-zero values to avoid log(0)
    p_xy_nz = p_xy[p_xy > 0]
    p_x_nz = p_x[np.sum(p_xy, axis=1) > 0]
    p_y_nz = p_y[np.sum(p_xy, axis=0) > 0]

    h_x = -np.sum(p_x_nz * np.log(p_x_nz))
    h_y = -np.sum(p_y_nz * np.log(p_y_nz))
    h_xy = -np.sum(p_xy_nz * np.log(p_xy_nz))

    mi = h_x + h_y - h_xy
    return max(0.0, mi)
