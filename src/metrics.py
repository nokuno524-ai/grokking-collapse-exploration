import torch
import torch.nn as nn
from collections import Counter
import numpy as np

def compute_fourier_concentration(model: nn.Module, top_k: int = 5) -> float:
    """
    Measure how concentrated the Fourier spectrum is on the top-k frequencies.
    High concentration → grokking has occurred (or is occurring).
    """
    if not hasattr(model, 'get_embedding_fourier_spectrum'):
        return 0.0
    spectrum = model.get_embedding_fourier_spectrum()  # (prime, d_model)
    # Average across embedding dimensions
    avg_spectrum = spectrum.mean(dim=1)  # (prime,)
    # Exclude DC component
    avg_spectrum = avg_spectrum[1:]
    total_energy = avg_spectrum.sum()
    if total_energy < 1e-10:
        return 0.0
    top_energy = avg_spectrum.topk(min(top_k, len(avg_spectrum))).values.sum()
    return (top_energy / total_energy).item()

def mode_collapse_score(predictions: torch.Tensor, prime: int) -> float:
    """
    Measures mode collapse.
    Returns 1.0 if all predictions are the same class, 0.0 if predictions are uniformly distributed.
    """
    if len(predictions) == 0:
        return 0.0
    counts = torch.bincount(predictions, minlength=prime).float()
    probs = counts / counts.sum()
    # Compute entropy
    entropy = -(probs * torch.log(probs + 1e-10)).sum()
    max_entropy = torch.log(torch.tensor(prime, dtype=torch.float))
    # 0 entropy -> score 1.0 (collapse)
    # max entropy -> score 0.0 (no collapse)
    score = 1.0 - (entropy / max_entropy)
    return score.item()

def kl_divergence_shift(predictions: torch.Tensor, targets: torch.Tensor, prime: int) -> float:
    """
    Measures distribution shift (KL divergence) between predictions and targets.
    """
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0
    pred_counts = torch.bincount(predictions, minlength=prime).float()
    target_counts = torch.bincount(targets, minlength=prime).float()

    p = pred_counts / pred_counts.sum()
    q = target_counts / target_counts.sum()

    # KL(P || Q) = sum P(x) log(P(x) / Q(x))
    # We add epsilon to Q to avoid division by zero
    kl = (p * torch.log((p + 1e-10) / (q + 1e-10))).sum()
    return kl.item()

def loss_of_complexity(model: nn.Module) -> float:
    """
    Measures the loss of complexity via the embedding effective rank.
    A lower rank implies loss of complexity.
    """
    if hasattr(model, 'get_embedding_rank'):
        return model.get_embedding_rank()
    return 0.0

def memorization_score(train_acc: float, test_acc: float) -> float:
    """
    Measures memorization: high train acc and low test acc = high memorization.
    Bounded between 0.0 and 1.0.
    """
    return max(0.0, train_acc - test_acc)
