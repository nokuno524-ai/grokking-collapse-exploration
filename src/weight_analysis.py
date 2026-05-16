import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple
import os
from pathlib import Path
from scipy.stats import wasserstein_distance

class WeightAnalyzer:
    """Analyzes model weights, tracks norms, and computes singular spectra."""

    @staticmethod
    def compute_weight_norms(model: nn.Module, p: Union[int, str] = 'fro') -> Dict[str, float]:
        """
        Computes the norm of the weights for each named parameter in the model.
        """
        norms = {}
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                norms[name] = torch.linalg.matrix_norm(param.data.float(), ord=p).item()
        return norms

    @staticmethod
    def track_norm_evolution(checkpoints_dir: Union[str, Path], model_class: callable, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Loads checkpoints from a directory and tracks weight norms over time.
        """
        dir_path = Path(checkpoints_dir)
        checkpoints = list(dir_path.glob("checkpoint_*.pt"))

        # Sort by step
        def get_step(path):
            try:
                return int(path.stem.split('_')[-1])
            except ValueError:
                return -1

        checkpoints.sort(key=get_step)

        data = []
        for ckpt_path in checkpoints:
            step = get_step(ckpt_path)
            if step == -1:
                continue

            ckpt = torch.load(ckpt_path, map_location="cpu")
            model = model_class(**config)
            model.load_state_dict(ckpt["model_state"])

            norms = WeightAnalyzer.compute_weight_norms(model)
            norms['step'] = step
            data.append(norms)

        return pd.DataFrame(data)

    @staticmethod
    def compute_effective_rank(weight_matrix: Union[torch.Tensor, np.ndarray], threshold: float = 0.99) -> int:
        """
        Computes the effective rank of a matrix based on singular values.
        It returns the number of singular values needed to explain 'threshold' fraction of the variance.
        """
        if isinstance(weight_matrix, np.ndarray):
            w = torch.from_numpy(weight_matrix).float()
        else:
            w = weight_matrix.float()

        if w.dim() > 2:
            w = w.view(w.size(0), -1)

        S = torch.linalg.svdvals(w)

        # Calculate explained variance
        S_squared = S ** 2
        total_variance = torch.sum(S_squared)

        if total_variance == 0:
            return 0

        explained_variance_ratio = torch.cumsum(S_squared, dim=0) / total_variance

        effective_rank = torch.searchsorted(explained_variance_ratio, threshold).item() + 1
        return effective_rank

    @staticmethod
    def compute_singular_spectrum(weight_matrix: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Returns the normalized singular values of a matrix.
        """
        if isinstance(weight_matrix, np.ndarray):
            w = torch.from_numpy(weight_matrix).float()
        else:
            w = weight_matrix.float()

        if w.dim() > 2:
            w = w.view(w.size(0), -1)

        S = torch.linalg.svdvals(w)
        S_norm = S / S.max() if S.max() > 0 else S

        return S_norm.cpu().numpy()

    @staticmethod
    def compare_weight_distributions(model_a: nn.Module, model_b: nn.Module) -> Dict[str, float]:
        """
        Compares the weight distributions of two models using Wasserstein distance.
        Models must have the same architecture.
        """
        distances = {}

        params_a = dict(model_a.named_parameters())
        params_b = dict(model_b.named_parameters())

        for name in params_a:
            if name in params_b and 'weight' in name:
                flat_a = params_a[name].data.cpu().numpy().flatten()
                flat_b = params_b[name].data.cpu().numpy().flatten()

                dist = wasserstein_distance(flat_a, flat_b)
                distances[name] = dist

        return distances
