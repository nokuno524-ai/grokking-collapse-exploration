"""
SVD Spectrum Analysis
Computes effective rank evolution and waterfall plots of eigenvalues across training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def compute_svd_spectrum(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Computes singular values of embedding, Transformer FFN, and output head.
    """
    spectra = {}

    with torch.no_grad():
        # Token embedding
        w_emb = model.token_embed.weight.detach()
        s_emb = torch.linalg.svdvals(w_emb)
        spectra["embedding"] = s_emb

        # Output head
        w_out = model.output_head.weight.detach()
        s_out = torch.linalg.svdvals(w_out)
        spectra["output_head"] = s_out

        # Transformer linear layers (first layer)
        layer = model.transformer.layers[0] if hasattr(model.transformer, 'layers') else list(model.transformer.children())[0]
        w_ffn1 = layer.linear1.weight.detach()
        s_ffn1 = torch.linalg.svdvals(w_ffn1)
        spectra["ffn1"] = s_ffn1

        w_ffn2 = layer.linear2.weight.detach()
        s_ffn2 = torch.linalg.svdvals(w_ffn2)
        spectra["ffn2"] = s_ffn2

    return spectra


def compute_effective_rank(singular_values: torch.Tensor) -> float:
    """
    Compute effective rank using Shannon entropy of normalized singular values.
    """
    s = singular_values / singular_values.sum()
    entropy = -(s * torch.log(s + 1e-10)).sum()
    return torch.exp(entropy).item()


def track_svd_evolution(checkpoints: List[str], device: torch.device) -> Dict[str, List[float]]:
    """
    Tracks effective rank of various layers over a list of checkpoint paths.
    """
    from src.model import ModularArithmeticTransformer

    history = {
        "embedding": [],
        "output_head": [],
        "ffn1": [],
        "ffn2": [],
        "steps": []
    }

    for ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            continue

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        step = ckpt.get("step", 0)
        history["steps"].append(step)

        # Initialize model (assuming standard dimensions, or load config from ckpt)
        config = ckpt.get("config", {})
        prime = config.get("prime", 59)
        d_model = config.get("d_model", 128)

        model = ModularArithmeticTransformer(prime=prime, d_model=d_model).to(device)
        model.load_state_dict(ckpt["model_state"])

        spectra = compute_svd_spectrum(model)
        for key in ["embedding", "output_head", "ffn1", "ffn2"]:
            rank = compute_effective_rank(spectra[key])
            history[key].append(rank)

    return history


def plot_svd_waterfall(singular_value_history: List[torch.Tensor], output_path: str):
    """
    Plot eigenvalue distribution over training steps (waterfall plot).
    singular_value_history: list of singular value tensors corresponding to different steps.
    """
    if not HAS_MATPLOTLIB:
        return

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    for i, s in enumerate(singular_value_history):
        plt.plot(s.cpu().numpy(), alpha=0.5, label=f"Step {i}")

    plt.yscale('log')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('SVD Spectrum Waterfall')
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def identify_rank_collapse_patterns(rank_history: Dict[str, List[float]], grokking_step: Optional[int]) -> Dict[str, str]:
    """
    Identify rank collapse patterns that correlate with grokking failure.
    Returns a dictionary summarizing the collapse behavior per layer.
    """
    patterns = {}
    for layer in ["embedding", "output_head", "ffn1", "ffn2"]:
        if layer not in rank_history or not rank_history[layer]:
            continue

        ranks = rank_history[layer]
        initial_rank = ranks[0]
        final_rank = ranks[-1]

        drop_ratio = final_rank / (initial_rank + 1e-8)

        if drop_ratio < 0.2:
            pattern = "Severe Rank Collapse"
        elif drop_ratio < 0.5:
            pattern = "Moderate Rank Decay"
        else:
            pattern = "Stable Rank"

        patterns[layer] = pattern

    return patterns
