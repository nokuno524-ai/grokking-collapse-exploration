import torch
import pytest
from src.circuit_discovery import activation_patching, path_patching, logit_attribution, compare_circuits
from src.model import ModularArithmeticTransformer

@pytest.fixture
def models():
    prime = 59
    m1 = ModularArithmeticTransformer(prime=prime, d_model=32)
    m2 = ModularArithmeticTransformer(prime=prime, d_model=32)
    return m1, m2

@pytest.fixture
def dummy_data():
    return torch.randint(0, 59, (4, 2))

def test_activation_patching(models, dummy_data):
    m_recv, m_donor = models

    orig_logits, patched_logits = activation_patching(m_recv, m_donor, 'token_embed', dummy_data)

    assert orig_logits.shape == patched_logits.shape
    assert orig_logits.shape == (4, 59)
    # They should likely be different since we patched with a differently initialized model
    # but not necessarily mathematically guaranteed if weights happen to be close,
    # though almost certain with random init.

def test_path_patching(models, dummy_data):
    m1, _ = models
    logits = path_patching(m1, head_idx=0, data=dummy_data)

    assert logits.shape == (4, 59)

def test_logit_attribution(models, dummy_data):
    m1, _ = models
    attr = logit_attribution(m1, dummy_data)

    assert 'token_embed_direct' in attr
    assert 'pos_embed_direct' in attr

    assert attr['token_embed_direct'].shape == (4, 59)
    assert attr['pos_embed_direct'].shape == (4, 59)

def test_compare_circuits(models):
    m1, m2 = models

    # Compare with itself
    diffs_self = compare_circuits(m1, m1)
    for v in diffs_self.values():
        assert v < 1e-5

    # Compare with another model
    diffs_other = compare_circuits(m1, m2)
    for k, v in diffs_other.items():
        if "bias" not in k and "norm" not in k and "ln" not in k:
            assert v > 0.0
