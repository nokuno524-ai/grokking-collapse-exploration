import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.probes.probe import train_linear_probe, collect_hidden

class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def test_full_pipeline_with_dummy_mlp(tmp_path):
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 100
    batch = torch.randn(n_samples, 2)

    # Create random labels not dependent on the batch features so layer1 can't easily guess it
    y = np.random.randint(0, 2, n_samples)

    model = DummyMLP()

    ckpt_path = tmp_path / "dummy_checkpoint.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    hidden = collect_hidden(ckpt_path, batch, is_test=True, test_model=model)

    h1 = hidden['layer1'].numpy()
    h2 = hidden['layer2'].numpy()

    # Plant perfectly in layer 2
    h2[y == 1, 0] = 100.0
    h2[y == 0, 0] = -100.0

    acc1, std1 = train_linear_probe(h1, y, k_fold=5)
    acc2, std2 = train_linear_probe(h2, y, k_fold=5)

    assert acc1 < 0.7, f"Layer 1 accuracy too high: {acc1}"
    assert acc2 > 0.9, f"Layer 2 accuracy too low: {acc2}"
