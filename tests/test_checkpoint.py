import torch
import os
import json
import pytest
from src.checkpoint import load_run
from src.model import ModularArithmeticTransformer

def test_load_run(tmp_path):
    # Create dummy run
    config = {"prime": 11, "d_model": 32, "n_heads": 2}
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    model = ModularArithmeticTransformer(prime=11, d_model=32, n_heads=2)
    torch.save(model.state_dict(), tmp_path / "checkpoint_0.pt")
    torch.save(model.state_dict(), tmp_path / "checkpoint_100.pt")

    # Test loading specific step
    loaded = load_run(tmp_path, step=0)
    assert loaded.prime == 11
    assert not loaded.training # Should be in eval mode

    # Test loading latest
    loaded_latest = load_run(tmp_path)
    assert loaded_latest.prime == 11

    # Test loading final model if it exists
    torch.save(model.state_dict(), tmp_path / "model.pt")
    loaded_final = load_run(tmp_path)
    assert loaded_final.prime == 11

    # Test missing checkpoint
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path, step=999)
