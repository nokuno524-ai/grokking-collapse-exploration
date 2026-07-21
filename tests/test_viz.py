import pytest
import os
import torch
import numpy as np

from viz.attention_evolution import load_attention_weights
from viz.circuit_analysis import calculate_induction_score, cluster_attention_heads
from viz.weight_analysis import extract_weight_norm_trajectory

from tests.generate_checkpoint import generate_checkpoint

DUMMY_CKPT = "tests/data/dummy_checkpoint.pt"

@pytest.fixture
def checkpoint_path():
    if not os.path.exists(DUMMY_CKPT):
        generate_checkpoint()
    return DUMMY_CKPT

def test_load_attention_weights(checkpoint_path):
    attn = load_attention_weights(checkpoint_path)
    assert attn is not None
    assert isinstance(attn, torch.Tensor)
    # Should be (n_heads, L, L)
    assert len(attn.shape) == 3
    assert attn.shape[0] == 4 # 4 heads in our config
    assert attn.shape[1] == 2 # sequence length 2

def test_calculate_induction_score(checkpoint_path):
    attn = load_attention_weights(checkpoint_path)
    scores = calculate_induction_score(attn)
    assert scores is not None
    assert isinstance(scores, torch.Tensor)
    assert scores.shape[0] == 4

def test_cluster_attention_heads(checkpoint_path):
    attn = load_attention_weights(checkpoint_path)
    # We need at least n_clusters samples
    if attn.shape[0] >= 2:
        clusters = cluster_attention_heads(attn, n_clusters=2)
        assert clusters is not None
        assert isinstance(clusters, np.ndarray)
        assert len(clusters) == 4

def test_extract_weight_norm_trajectory(checkpoint_path):
    traj = extract_weight_norm_trajectory([checkpoint_path, checkpoint_path], [0, 1000])
    assert traj is not None
    assert 'total_norm' in traj
    assert 'token_embed.weight' in traj
    assert len(traj['total_norm']) == 2

def test_compute_hessian_max_eigenvalue(checkpoint_path):
    from viz.weight_analysis import compute_hessian_max_eigenvalue
    from src.model import ModularArithmeticTransformer
    import torch.nn as nn

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = ModularArithmeticTransformer()
    model.load_state_dict(checkpoint['model_state'])

    x = torch.randint(0, 59, (4, 2))
    y = (x[:, 0] + x[:, 1]) % 59
    loss_fn = nn.CrossEntropyLoss()

    eig = compute_hessian_max_eigenvalue(model, loss_fn, (x, y), max_iter=2)
    assert isinstance(eig, float)

def test_plot_loss_landscape_contour(checkpoint_path):
    from viz.weight_analysis import plot_loss_landscape_contour
    from src.model import ModularArithmeticTransformer
    import torch.nn as nn

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = ModularArithmeticTransformer()
    model.load_state_dict(checkpoint['model_state'])

    x = torch.randint(0, 59, (4, 2))
    y = (x[:, 0] + x[:, 1]) % 59
    loss_fn = nn.CrossEntropyLoss()

    loss_grid = plot_loss_landscape_contour(model, loss_fn, (x, y), grid_size=3)
    assert isinstance(loss_grid, np.ndarray)
    assert loss_grid.shape == (3, 3)