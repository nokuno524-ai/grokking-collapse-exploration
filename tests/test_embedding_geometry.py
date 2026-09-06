import torch
import numpy as np
from src.analysis.embedding_geometry import (
    get_effective_rank,
    get_pca_spectrum,
    get_pairwise_cosine_similarity,
    get_hidden_states_pca
)

def test_effective_rank():
    # Identity matrix should have effective rank N
    W = torch.eye(10)
    rank = get_effective_rank(W)
    assert abs(rank - 10.0) < 1e-4

def test_pca_spectrum():
    # Diagonal matrix with known singular values
    W = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    spectrum = get_pca_spectrum(W)
    assert len(spectrum) == 3
    assert abs(spectrum[0] - 3.0) < 1e-4
    assert abs(spectrum[1] - 2.0) < 1e-4
    assert abs(spectrum[2] - 1.0) < 1e-4

def test_pairwise_cosine_similarity():
    # Orthogonal vectors should have similarity 0
    W = torch.eye(3)
    sims = get_pairwise_cosine_similarity(W)
    assert len(sims) == 3  # (3 * 2) / 2 = 3
    assert np.allclose(sims, 0.0)

    # Identical vectors should have similarity 1
    W = torch.ones((3, 4))
    sims = get_pairwise_cosine_similarity(W)
    assert np.allclose(sims, 1.0)

def test_hidden_states_pca():
    hidden_states = torch.randn(10, 16)
    proj = get_hidden_states_pca(hidden_states)
    assert proj.shape == (10, 2)

    # Test fallback on too few samples
    hidden_states_small = torch.randn(1, 16)
    proj_small = get_hidden_states_pca(hidden_states_small)
    assert proj_small.shape == (1, 2)
    assert np.allclose(proj_small, 0.0)

from unittest.mock import patch, MagicMock
from pathlib import Path
from src.analysis.embedding_geometry import process_checkpoint, compare_trajectories
import json

def test_process_checkpoint(tmp_path):
    from src.model import ModularArithmeticTransformer
    # Setup mock checkpoint
    model = ModularArithmeticTransformer(prime=59)
    ckpt = {
        "step": 100,
        "config": {"prime": 59},
        "model_state": model.state_dict()
    }
    ckpt_path = tmp_path / "checkpoint_100.pt"
    torch.save(ckpt, ckpt_path)

    eval_batch = torch.randint(0, 59, (10, 2))
    metrics = process_checkpoint(ckpt_path, eval_batch, torch.device("cpu"))

    assert metrics["step"] == 100
    assert "effective_rank" in metrics
    assert "pca_spectrum" in metrics
    assert "cosine_similarity_histogram" in metrics
    assert "hidden_states_pca" in metrics
    assert len(metrics["hidden_states_pca"]) == 10

def test_compare_trajectories(tmp_path):
    pure_dir = tmp_path / "pure"
    contam_dir = tmp_path / "contam"
    pure_dir.mkdir()
    contam_dir.mkdir()

    pure_file = pure_dir / "embedding_geometry.jsonl"
    contam_file = contam_dir / "embedding_geometry.jsonl"

    with open(pure_file, "w") as f:
        f.write(json.dumps({"step": 1}) + "\n")
    with open(contam_file, "w") as f:
        f.write(json.dumps({"step": 1}) + "\n")

    pure_traj, contam_traj = compare_trajectories(pure_dir, contam_dir)
    assert len(pure_traj) == 1
    assert pure_traj[0]["step"] == 1
    assert len(contam_traj) == 1
