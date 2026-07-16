import torch
import pytest
from src.model import ModularArithmeticTransformer
from analysis.gradient_flow import approximate_gradients, identify_gradient_starvation, track_gradient_norms, measure_gradient_noise_scale

def test_approximate_gradients():
    m1 = ModularArithmeticTransformer()
    m2 = ModularArithmeticTransformer()

    # Force m2 weights to exactly m1 weights + 0.1
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        p2.data = p1.data + 0.1

    grads = approximate_gradients(m1, m2)

    for name, grad in grads.items():
        assert torch.allclose(grad, torch.ones_like(grad) * 0.1)

def test_identify_gradient_starvation():
    grads_pure = {
        "embed.weight": torch.ones(10) * 1.0,
        "fc.weight": torch.ones(10) * 1.0
    }

    grads_collapsed = {
        "embed.weight": torch.ones(10) * 0.01,  # Starved
        "fc.weight": torch.ones(10) * 0.9      # Not starved
    }

    starved = identify_gradient_starvation(grads_pure, grads_collapsed, threshold_ratio=0.1)
    assert "embed.weight" in starved
    assert "fc.weight" not in starved

def test_track_gradient_norms():
    m1 = ModularArithmeticTransformer()
    m2 = ModularArithmeticTransformer()

    for param in m2.parameters():
        param.data += 0.1

    ckpt1 = {'model_state': m1.state_dict()}
    ckpt2 = {'model_state': m2.state_dict()}

    norms_dict = track_gradient_norms([ckpt1, ckpt2])
    assert len(norms_dict) > 0
    for name, norms in norms_dict.items():
        assert len(norms) == 1
        assert norms[0] > 0

def test_measure_gradient_noise_scale():
    model = ModularArithmeticTransformer(d_model=16, n_heads=1, d_ff=32)
    x = torch.randint(0, 59, (4, 2))
    y = torch.randint(0, 59, (4,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    criterion = torch.nn.CrossEntropyLoss()

    gns = measure_gradient_noise_scale(model, loader, criterion)

    assert isinstance(gns, float)
    assert gns >= 0.0
