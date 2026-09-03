import torch
import pytest
import tempfile
from pathlib import Path
import json

from src.transplant.circuits import swap_attention_head, swap_mlp
from src.transplant.run_transplants import random_basis_swap, run_transplants_for_pair
from src.model import ModularArithmeticTransformer

def test_random_basis_swap():
    rng = torch.Generator().manual_seed(42)
    w = torch.randn(10, 10)
    w_swap = random_basis_swap(w, rng)
    assert w.shape == w_swap.shape
    assert not torch.allclose(w, w_swap)

def test_random_basis_swap_non_square():
    rng = torch.Generator().manual_seed(42)
    # Test embedding shape
    w = torch.randn(59, 128)
    w_swap = random_basis_swap(w, rng)
    assert w.shape == w_swap.shape

def test_identical_weights_no_op():
    d_model = 128
    n_heads = 4
    model = ModularArithmeticTransformer(d_model=d_model, n_heads=n_heads, n_layers=1)
    sd = model.state_dict()

    # Transplanting onto itself should yield identical state dict
    patched_sd = swap_attention_head(sd, sd, 0, 0, n_heads, d_model)

    for k in sd:
        assert torch.allclose(sd[k], patched_sd[k]), f"Mismatch in {k} after identity transplant"

def test_transplant_permuted_head():
    d_model = 128
    n_heads = 4
    model = ModularArithmeticTransformer(d_model=d_model, n_heads=n_heads, n_layers=1)
    base_sd = model.state_dict()

    # Get base outputs
    x = torch.randint(0, 59, (4, 2))
    base_out = model(x)

    donor_sd = {k: v.clone() for k, v in base_sd.items()}
    # Perturb the donor's head 0 in layer 0 significantly
    head_dim = d_model // n_heads
    donor_sd["transformer.layers.0.self_attn.in_proj_weight"][:head_dim, :] += 10.0

    patched_sd = swap_attention_head(base_sd, donor_sd, 0, 0, n_heads, d_model)

    # Load patched sd and get new outputs
    patched_model = ModularArithmeticTransformer(d_model=d_model, n_heads=n_heads, n_layers=1)
    patched_model.load_state_dict(patched_sd, strict=True)
    patched_out = patched_model(x)

    # Base should not be mutated in memory (outputs should differ from patched)
    assert not torch.allclose(base_out, patched_out)

    # Also verify internal structures
    assert not torch.allclose(base_sd["transformer.layers.0.self_attn.in_proj_weight"], patched_sd["transformer.layers.0.self_attn.in_proj_weight"])
    # But other heads should be identical
    assert torch.allclose(
        base_sd["transformer.layers.0.self_attn.in_proj_weight"][head_dim:, :],
        patched_sd["transformer.layers.0.self_attn.in_proj_weight"][head_dim:, :]
    )

def test_matrix_runner_stub():
    """Test the matrix runner end-to-end on a tiny stub config."""
    # Create two temporary directories representing runs
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        dir1, dir2 = Path(d1), Path(d2)

        cfg = {
            "prime": 7,
            "d_model": 16,
            "n_heads": 2,
            "d_ff": 32,
            "n_layers": 1,
            "seed": 42,
            "train_fraction": 0.5,
            "collapse_level": 0.0,
            "collapse_severity": 0.0,
            "noise_fraction": 0.0
        }

        model1 = ModularArithmeticTransformer(**{k: v for k, v in cfg.items() if k in ["prime", "d_model", "n_heads", "d_ff", "n_layers"]})
        model2 = ModularArithmeticTransformer(**{k: v for k, v in cfg.items() if k in ["prime", "d_model", "n_heads", "d_ff", "n_layers"]})

        torch.save({"model_state": model1.state_dict(), "config": cfg}, dir1 / "checkpoint_100.pt")
        torch.save({"model_state": model2.state_dict(), "config": cfg}, dir2 / "checkpoint_100.pt")

        with open(dir1 / "results.json", "w") as f:
            json.dump({"config": cfg}, f)
        with open(dir2 / "results.json", "w") as f:
            json.dump({"config": cfg}, f)

        # Run transplant
        device = torch.device("cpu")
        results = run_transplants_for_pair(dir1, dir2, "pure", "severe", device)

        # Expect 2 heads, 1 mlp, 2 lns = 5 results
        assert len(results) == 5
        assert results[0].donor_condition == "pure"
        assert results[0].recipient_condition == "severe"
