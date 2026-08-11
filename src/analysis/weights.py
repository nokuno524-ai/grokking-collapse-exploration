import torch
import torch.nn as nn
from scipy.stats import kurtosis, skew
import numpy as np
from typing import Dict, Any

def compute_weight_stats(model: nn.Module) -> Dict[str, Any]:
    """Compute weight distribution statistics including kurtosis, skewness, sparsity, and SVD rank."""
    stats = {}

    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            w_np = param.detach().cpu().numpy().flatten()
            if len(w_np) > 1:
                stats[f"{name}_kurtosis"] = float(kurtosis(w_np))
                stats[f"{name}_skewness"] = float(skew(w_np))
                stats[f"{name}_sparsity"] = float(np.mean(np.abs(w_np) < 1e-5))

            # SVD rank for 2D weight matrices
            if len(param.shape) == 2:
                s = torch.linalg.svdvals(param.detach().float())
                s_sum = s.sum().item()
                if s_sum > 1e-10:
                    s_norm = s / s_sum
                    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum().item()
                    stats[f"{name}_svd_rank"] = np.exp(entropy)
                else:
                    stats[f"{name}_svd_rank"] = 1.0

    return stats
