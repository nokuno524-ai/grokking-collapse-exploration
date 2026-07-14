import torch
import pytest
from src.model import ModularArithmeticTransformer
from analysis.attention_evolution import get_attention_entropy, classify_head_context_matching

def test_attention_entropy():
    model = ModularArithmeticTransformer(prime=7) # small prime for speed
    entropy = get_attention_entropy(model, prime=7)

    # check shape
    assert entropy.shape == (model.n_heads, 2) # n_heads x seq_len
    # entropy of uniform distribution over 2 items is ln(2) = 0.693
    # should be less than or equal to this
    assert torch.all(entropy <= torch.log(torch.tensor(2.0)) + 1e-4)
    assert torch.all(entropy >= 0)

def test_context_matching():
    model = ModularArithmeticTransformer(prime=7)
    weights = classify_head_context_matching(model, prime=7)

    # shape n_heads x seq_len x seq_len
    assert weights.shape == (model.n_heads, 2, 2)
    # weights sum to 1 over last dim
    assert torch.allclose(weights.sum(dim=-1), torch.ones(model.n_heads, 2), atol=1e-4)
