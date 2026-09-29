import pytest
import torch
import json
from pathlib import Path
from src.checkpoint import load_checkpoint, list_checkpoints, load_run_config

def test_load_run_config(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Missing file
    assert load_run_config(run_dir) is None

    # Valid file
    res_path = run_dir / "results.json"
    res_path.write_text(json.dumps({"config": {"prime": 59}}))
    assert load_run_config(run_dir) == {"prime": 59}

    # Invalid json
    res_path.write_text("{invalid")
    assert load_run_config(run_dir) is None

def test_list_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "checkpoint_100.pt").touch()
    (run_dir / "checkpoint_0.pt").touch()
    (run_dir / "checkpoint_50.pt").touch()

    ckpts = list_checkpoints(run_dir)
    assert len(ckpts) == 3
    assert ckpts[0][0] == 0
    assert ckpts[1][0] == 50
    assert ckpts[2][0] == 100

def test_load_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # No checkpoints
    with pytest.raises(FileNotFoundError):
        load_checkpoint(run_dir)

    # Valid checkpoint
    ckpt_path = run_dir / "checkpoint_100.pt"

    sd = {"token_embed.weight": torch.randn(59, 128)}
    torch.save({"model_state": sd, "config": {"prime": 59}}, ckpt_path)

    loaded_sd, cfg = load_checkpoint(run_dir)
    assert "token_embed.weight" in loaded_sd
    assert cfg["prime"] == 59

    # Shape mismatch error
    with pytest.raises(ValueError):
        load_checkpoint(run_dir, strict_config_match={"prime": 97})
