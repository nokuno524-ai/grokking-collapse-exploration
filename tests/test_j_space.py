import torch
import pytest
from src.model import ModularArithmeticTransformer
from analysis.j_space_probe import get_jacobian, get_j_space_svd, causal_intervention_j_space

def test_jacobian():
    model = ModularArithmeticTransformer(prime=7)
    J = get_jacobian(model, prime=7)
    assert J.shape == (7, model.d_model)

def test_j_space_svd():
    model = ModularArithmeticTransformer(prime=7)
    U, S, Vh = get_j_space_svd(model, prime=7)

    assert U.shape == (7, 7)
    assert S.shape == (7,)
    assert Vh.shape == (7, model.d_model)

    # singular values should be non-negative and sorted
    assert torch.all(S >= -1e-5)
    assert torch.all(S[:-1] >= S[1:])

def test_causal_intervention():
    model = ModularArithmeticTransformer(prime=7)
    results = causal_intervention_j_space(model, prime=7)

    assert 'base_acc' in results
    assert 'interv_acc_top_5_removed' in results
    assert 0.0 <= results['base_acc'] <= 1.0
    assert 0.0 <= results['interv_acc_top_5_removed'] <= 1.0
