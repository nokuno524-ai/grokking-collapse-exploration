import torch
import math
import numpy as np
from src.model import ModularArithmeticTransformer
from src.analysis.attention import compute_attention_metrics
from src.analysis.kmeans_probe import compute_kmeans_signature, signature_from_tensors
from src.analysis.attention import calculate_metrics_from_weights

def get_deterministic_model():
    torch.manual_seed(42)
    model = ModularArithmeticTransformer(d_model=16, n_heads=2, d_ff=32)
    model.eval()
    return model

def get_deterministic_dataloader():
    torch.manual_seed(42)
    # 4 examples, length 2 sequence
    inputs = torch.randint(0, 59, (4, 2))
    targets = (inputs[:, 0] + inputs[:, 1]) % 59
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    return torch.utils.data.DataLoader(dataset, batch_size=2)

def test_compute_attention_metrics():
    model = get_deterministic_model()
    loader = get_deterministic_dataloader()

    metrics = compute_attention_metrics(model, loader)

    assert "entropy" in metrics
    assert "specialization" in metrics

    # Check shape: num_heads = 2
    assert metrics["entropy"].shape == (2,)
    assert metrics["specialization"].shape == (2,)

    # Since model is initialized deterministically, values should be stable
    # The uniform distribution for sequence length 2 has entropy -2 * 0.5 * ln(0.5) = 0.693
    # Check that entropy is not negative and not higher than uniform
    for e in metrics["entropy"]:
        assert 0.0 <= e <= 0.7

def test_kmeans_signature_probe():
    model = get_deterministic_model()
    loader = get_deterministic_dataloader()

    signature = compute_kmeans_signature(model, loader)

    assert signature.shape == (2,)

    # Signature is correlation, so should be between -1.0 and 1.0
    for s in signature:
        assert -1.0 <= s <= 1.0

def test_kmeans_signature_synthetic():
    h = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]],

        [[1.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0]],
    ])

    dist = torch.norm(h.unsqueeze(2) - h.unsqueeze(1), dim=-1)

    attn = torch.exp(-dist)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    attn_weights = attn.unsqueeze(1) # Add head dimension

    sig = signature_from_tensors(h, attn_weights)
    assert len(sig) == 1
    assert math.isclose(sig[0], 0.7144, abs_tol=1e-4)


def test_attention_metrics_synthetic():
    # Identity matrix attention (1.0 on diagonal) - highly specialized, zero entropy
    attn_weights = torch.eye(4).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
    entropy, spec = calculate_metrics_from_weights(attn_weights)
    assert math.isclose(entropy.item(), 0.0, abs_tol=1e-5)

    # Uniform attention - max entropy, low specialization
    attn_weights_uniform = torch.ones(1, 1, 4, 4) / 4.0
    entropy_u, spec_u = calculate_metrics_from_weights(attn_weights_uniform)
    assert math.isclose(entropy_u.item(), -math.log(0.25), abs_tol=1e-5)
    assert math.isclose(spec_u.item(), 0.0, abs_tol=1e-5)
