import torch
import numpy as np
from pathlib import Path
from src.transplant.robustness import check_robustness, compute_head_importance
from src.model import ModularArithmeticTransformer
from src.data import DatasetConfig

def test_compute_head_importance(tmp_path):
    device = torch.device("cpu")
    config = DatasetConfig(prime=59)

    pure = ModularArithmeticTransformer(prime=59)
    contam = ModularArithmeticTransformer(prime=59)

    # Just a dummy eval batch
    inputs = torch.randint(0, 59, (10, 2))
    targets = (inputs[:, 0] + inputs[:, 1]) % 59
    test_ds = torch.utils.data.TensorDataset(inputs, targets)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

    imp = compute_head_importance(pure, contam, test_loader, device, config, True)

    assert imp.shape == (1, 4)
    # With random initialized models, importance will be very close to 0
    assert np.all(np.abs(imp) < 1.0)

def test_check_robustness(tmp_path):
    pure_dir = tmp_path / "pure"
    contam_dir = tmp_path / "contam"
    pure_dir.mkdir()
    contam_dir.mkdir()

    pure_model = ModularArithmeticTransformer(prime=59)
    contam_model = ModularArithmeticTransformer(prime=59)

    torch.save({
        "step": 1000,
        "config": {"prime": 59},
        "model_state": pure_model.state_dict()
    }, pure_dir / "checkpoint_1000.pt")

    torch.save({
        "step": 1000,
        "config": {"prime": 59},
        "model_state": contam_model.state_dict()
    }, contam_dir / "checkpoint_1000.pt")

    res = check_robustness(pure_dir, contam_dir, torch.device("cpu"), seed_variations=[42, 100])

    assert "mean_correlation" in res
    assert "min_correlation" in res
    # With identical initializations for pure and contam for all seeds,
    # the correlation could be nan and fallback to 0 or 1.
    assert isinstance(res["mean_correlation"], float)
