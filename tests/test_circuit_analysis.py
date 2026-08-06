import torch
from src.model import ModularArithmeticTransformer
from src.analysis.circuit_analysis import activation_patch, discover_circuits, head_importance_scores

def test_activation_patch():
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4)
    clean_input = torch.randint(0, 59, (4, 2))
    corrupted_input = torch.randint(0, 59, (4, 2))

    # Patching at layer 0 in a 1-layer model should exactly match clean outputs
    clean_logits, patched_logits = activation_patch(model, clean_input, corrupted_input, [0])
    assert torch.allclose(clean_logits, patched_logits, atol=1e-4)

def test_discover_circuits():
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4)
    dataset = torch.utils.data.TensorDataset(torch.randint(0, 59, (32, 2)), torch.randint(0, 59, (32,)))

    scores = discover_circuits(model, dataset)
    assert 'layer_0' in scores
    assert isinstance(scores['layer_0'], float)

def test_head_importance_scores():
    model = ModularArithmeticTransformer(prime=59, d_model=128, n_heads=4)
    dataset = torch.utils.data.TensorDataset(torch.randint(0, 59, (32, 2)), torch.randint(0, 59, (32,)))

    importance = head_importance_scores(model, dataset)
    assert importance.shape == (4,)
    assert torch.all(importance >= 0)
    assert torch.allclose(importance.sum(), torch.tensor(1.0), atol=1e-4)
