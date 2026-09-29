import json
import torch
from pathlib import Path
from src.transplant.sweep import run_transplant_matrix

def test_transplant_matrix(tmp_path):
    # Setup mock checkpoint directories
    base_dir = tmp_path / "results"

    conds = ["pure", "collapse"]

    for cond in conds:
        cond_dir = base_dir / cond
        cond_dir.mkdir(parents=True)

        # Fake config
        cfg = {
            "prime": 7,
            "d_model": 16,
            "n_heads": 2,
            "d_ff": 32,
            "n_layers": 1,
            "train_fraction": 0.5,
            "seed": 42
        }

        # Generate dummy model to get state dict
        from src.model import ModularArithmeticTransformer
        model = ModularArithmeticTransformer(
            prime=7, d_model=16, n_heads=2, d_ff=32, n_layers=1
        )

        # Modify weights slightly to differentiate
        if cond == "pure":
            for p in model.parameters():
                p.data.fill_(0.1)
        else:
            for p in model.parameters():
                p.data.fill_(0.9)

        # Save mock checkpoint
        ckpt = {
            "model_state": model.state_dict(),
            "config": cfg,
            "step": 10
        }
        torch.save(ckpt, cond_dir / "checkpoint_10.pt")

    out_file = tmp_path / "matrix_output.json"

    # Run the sweep (use linear1 as test component)
    res = run_transplant_matrix(
        base_dir=base_dir,
        components_to_test=["linear1", "head_0_0"],
        conditions=conds,
        device=torch.device("cpu"),
        batch_size=4,
        output_file=out_file
    )

    # Check outputs
    assert "baselines" in res
    assert "results" in res

    assert "pure" in res["baselines"]
    assert "collapse" in res["baselines"]

    assert len(res["results"]) == len(conds) * len(conds) * 2 # 2 conditions x 2 conditions x 2 components

    assert out_file.exists()

    with open(out_file) as f:
        data = json.load(f)
        assert len(data["results"]) == 8
