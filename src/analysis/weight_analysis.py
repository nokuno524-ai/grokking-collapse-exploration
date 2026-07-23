import torch
import torch.nn as nn
from typing import Dict, List

def compute_weight_norm_trajectories(model_checkpoints: List[str]) -> List[float]:
    """Computes total weight norm trajectory across checkpoints."""
    norms = []
    for ckpt_path in model_checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state"]
        total_norm = 0.0
        for name, tensor in state_dict.items():
            total_norm += tensor.norm().item() ** 2
        norms.append(total_norm ** 0.5)
    return norms

def compute_layerwise_statistics(model: nn.Module) -> Dict[str, dict]:
    """Computes statistics (norm, mean, std) for each layer in the model."""
    stats = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        stats[name] = {
            "norm": param.norm().item(),
            "mean": param.mean().item(),
            "std": param.std().item()
        }
    return stats
