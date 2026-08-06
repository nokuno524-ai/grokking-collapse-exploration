import torch
from src.model import ModularArithmeticTransformer
from src.analysis.comparisons import cka_similarity, svcca_similarity, specialization_score

def test_cka_similarity():
    a = torch.randn(100, 64)
    b = a.clone()

    cka = cka_similarity(a, b)
    assert torch.isclose(torch.tensor(cka), torch.tensor(1.0), atol=1e-4)

    c = torch.randn(100, 64)
    cka_diff = cka_similarity(a, c)
    assert cka_diff < 1.0

def test_svcca_similarity():
    a = torch.randn(100, 64)
    b = a.clone()

    svcca = svcca_similarity(a, b)
    assert torch.isclose(torch.tensor(svcca), torch.tensor(1.0), atol=1e-4)

def test_specialization_score():
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4)
    dataset = torch.utils.data.TensorDataset(torch.randint(0, 59, (32, 2)), torch.randint(0, 59, (32,)))

    score = specialization_score(model, dataset)
    assert 0.0 <= score <= 1.0
