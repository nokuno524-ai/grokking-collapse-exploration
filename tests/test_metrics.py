import pytest
import torch
import math

# We can import from our newly created experiment scripts or reimplement simple logic for testing
from experiments.phase2_collapse_vs_noise import compute_kl_divergence, compute_information_content
from experiments.phase2_weight_analysis import compute_effective_rank
from experiments.phase2_fourier_analysis import analyze_fourier
from src.model import ModularArithmeticTransformer

def test_compute_effective_rank():
    # Identity matrix has equal singular values => max entropy
    W_id = torch.eye(10)
    rank_id = compute_effective_rank(W_id)
    # Entropy should be log(10), effective rank = exp(log(10)) = 10
    assert torch.isclose(torch.tensor(rank_id), torch.tensor(10.0), atol=1e-3)

    # Rank 1 matrix
    W_1 = torch.ones(10, 10)
    rank_1 = compute_effective_rank(W_1)
    # Effective rank should be close to 1
    assert torch.isclose(torch.tensor(rank_1), torch.tensor(1.0), atol=1e-3)

def test_compute_kl_divergence():
    p = {0: 0.5, 1: 0.5}
    q = {0: 0.5, 1: 0.5}
    assert compute_kl_divergence(p, q) == 0.0

    q2 = {0: 0.8, 1: 0.2}
    # KL(P||Q) = 0.5*log(0.5/0.8) + 0.5*log(0.5/0.2) = 0.5 * (-0.47 + 0.916) = 0.5 * 0.446 = 0.223
    kl = compute_kl_divergence(p, q2)
    assert kl > 0.0

def test_compute_information_content():
    # Uniform distribution over prime=59
    prime = 59
    uniform_probs = {i: 1.0/prime for i in range(prime)}
    info = compute_information_content(uniform_probs, prime)
    # Uniform distribution has max entropy, so information content should be 0 bits
    assert math.isclose(info, 0.0, abs_tol=1e-5)

    # Deterministic (one value)
    det_probs = {0: 1.0}
    info_det = compute_information_content(det_probs, prime)
    # Entropy is 0, so info content should be log2(prime)
    assert math.isclose(info_det, math.log2(prime), abs_tol=1e-5)

def test_analyze_fourier():
    model = ModularArithmeticTransformer(prime=59)
    concentration, spectrum = analyze_fourier(model, top_k=5)
    assert 0.0 <= concentration <= 1.0
    # Spectrum should exclude DC (so length is 58)
    assert len(spectrum) == 58
