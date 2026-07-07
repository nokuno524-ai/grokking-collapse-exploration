import torch
import torch.nn as nn
from scipy.stats import entropy
import numpy as np
from typing import Dict, List, Tuple
import os

def compute_effective_rank(weight: torch.Tensor) -> float:
    """
    Computes effective rank of a weight matrix using SVD Shannon entropy.
    Effective rank is exp(H(s)) where H is the Shannon entropy of normalized singular values.
    """
    if weight.ndim != 2:
        weight = weight.view(weight.size(0), -1)

    s = torch.linalg.svdvals(weight).detach().cpu().numpy()
    s_norm = s / (s.sum() + 1e-10)

    # Calculate Shannon entropy
    h = entropy(s_norm + 1e-10)

    return float(np.exp(h))

def compute_cosine_similarity(w1: torch.Tensor, w2: torch.Tensor) -> float:
    """
    Computes cosine similarity between two weight matrices (flattened into vectors).
    """
    w1_flat = w1.detach().view(-1)
    w2_flat = w2.detach().view(-1)

    return float(torch.nn.functional.cosine_similarity(w1_flat, w2_flat, dim=0))

def track_weight_evolution(model_checkpoints: List[nn.Module], layer_names: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Tracks effective rank and cosine similarity across consecutive checkpoints.
    Returns dict: {layer_name: {'effective_rank': [..], 'cosine_sim': [..]}}
    """
    results = {name: {'effective_rank': [], 'cosine_sim': []} for name in layer_names}

    for i, model in enumerate(model_checkpoints):
        state_dict = model.state_dict()

        for name in layer_names:
            if name not in state_dict:
                continue

            weight = state_dict[name]
            rank = compute_effective_rank(weight)
            results[name]['effective_rank'].append(rank)

            if i > 0:
                prev_weight = model_checkpoints[i-1].state_dict()[name]
                sim = compute_cosine_similarity(prev_weight, weight)
                results[name]['cosine_sim'].append(sim)
            else:
                results[name]['cosine_sim'].append(1.0) # Self similarity at step 0

    return results

def detect_phase_transitions(cosine_sims: List[float], steps: List[int], threshold: float = 0.95) -> List[int]:
    """
    Identifies phase transition steps where weights change most rapidly
    (i.e., cosine similarity drops below threshold).
    """
    transition_steps = []

    for sim, step in zip(cosine_sims, steps):
        if sim < threshold:
            transition_steps.append(step)

    return transition_steps
