import pytest
import torch
import torch.nn as nn
from src.model import ModularArithmeticTransformer
from src.transplant.transplant_rescue import patch_state_dict, freeze_component
from src.transplant.importance import compute_weight_norms, score_components
from src.transplant.atlas import build_component_state_dict_patch
from src.analysis.atlas_analysis import plot_importance_vs_rescue
from pathlib import Path
import json
import tempfile

def test_remapping_logic_identical():
    """Test remapping logic with identical architectures."""
    model_a = ModularArithmeticTransformer()
    model_b = ModularArithmeticTransformer()

    # Change a weight in model_b
    with torch.no_grad():
        model_b.transformer.layers[0].linear1.weight.fill_(1.0)

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    patched_sd = patch_state_dict(sd_a, sd_b, "linear1")

    # Check it patched correctly
    assert torch.all(patched_sd['transformer.layers.0.linear1.weight'] == 1.0)
    # Check it didn't patch everything
    assert not torch.all(patched_sd['transformer.layers.0.linear2.weight'] == sd_b['transformer.layers.0.linear2.weight'])

def test_remapping_logic_mismatched():
    """Test remapping logic throws error on mismatched shapes."""
    model_a = ModularArithmeticTransformer(d_model=128)
    model_b = ModularArithmeticTransformer(d_model=256)

    with pytest.raises(ValueError, match="shape mismatch"):
        patch_state_dict(model_a.state_dict(), model_b.state_dict(), "linear1")

def test_dtype_device_casting():
    """Test that patch_state_dict correctly casts to recipient dtype/device."""
    model_a = ModularArithmeticTransformer().eval() # Base (CPU, float32)
    model_b = ModularArithmeticTransformer().eval().to(torch.float64) # Donor (CPU, float64)

    patched_sd = patch_state_dict(model_a.state_dict(), model_b.state_dict(), "linear1")

    assert patched_sd['transformer.layers.0.linear1.weight'].dtype == torch.float32

def test_hook_symmetry():
    """Test that activation patching runs without breaking the model permanently."""
    model = ModularArithmeticTransformer()
    # Mock data
    x_clean = torch.randint(0, 59, (4, 2))
    y_clean = (x_clean[:, 0] + x_clean[:, 1]) % 59
    x_corr = torch.randint(0, 59, (4, 2))
    y_corr = (x_corr[:, 0] + x_corr[:, 1]) % 59

    scores = score_components(model, (x_clean, y_clean), (x_corr, y_corr))

    assert len(scores) > 0
    assert any(s.component_type == 'head' for s in scores)
    assert any(s.component_type == 'mlp' for s in scores)

def test_atlas_aggregation():
    """Test atlas results aggregation and plotting."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "results.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"component": "L0_H0", "donor_importance": 0.5, "zero_shot_test_acc": 0.2, "rescue_test_acc": 0.8}) + "\n")
            f.write(json.dumps({"component": "L0_H1", "donor_importance": 0.1, "zero_shot_test_acc": 0.1, "rescue_test_acc": 0.3}) + "\n")

        from src.analysis.atlas_analysis import read_atlas_results
        res = read_atlas_results(path)
        assert len(res) == 2

        out_path = Path(d) / "plot.png"
        plot_importance_vs_rescue([res], out_path)

        assert out_path.exists()
