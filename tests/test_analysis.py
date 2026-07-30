import pytest
import numpy as np
import torch
import torch.nn as nn
from src.analysis.statistics import (
    bootstrap_ci,
    permutation_test_grokking,
    benjamini_hochberg,
    cohens_d
)
from src.analysis.weight_analysis import (
    get_weight_norms,
    compute_effective_rank,
    measure_weight_sparsity,
    track_gradient_norms,
    compute_hessian_max_eigenvalue
)
from src.analysis.interpretability import (
    get_attention_head_attributions,
    activation_patching,
    get_head_saliency_maps,
    track_fourier_coefficients
)
from src.analysis.dashboard import generate_comparison_dashboard

# Setup dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = 2
        self.embed = nn.Embedding(10, 16)
        self.mha = nn.MultiheadAttention(16, self.n_heads, batch_first=True)
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 2)

    def forward(self, x):
        h = self.embed(x)
        h, _ = self.mha(h, h, h)
        h = h.mean(dim=1)
        h = torch.relu(self.linear1(h))
        return self.linear2(h)

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x = torch.randint(0, 10, (100, 5))
        self.y = torch.randint(0, 2, (100,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def test_statistics():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stat, lower, upper = bootstrap_ci(data, num_bootstrap=100)
    assert np.isclose(stat, 3.0)
    assert lower >= 1.0 and upper <= 5.0

    a = np.array([1000, 1100, 1050])
    b = np.array([2000, 2100, 2050])
    diff, p_val = permutation_test_grokking(a, b, num_permutations=100)
    assert np.isclose(diff, -1000.0)
    assert p_val >= 0.0 and p_val <= 1.0

    p_values = [0.01, 0.04, 0.03, 0.001]
    sig, adj = benjamini_hochberg(p_values, fdr_level=0.05)
    assert len(sig) == 4
    assert sig[3] is True  # np.bool_ check not needed if we converted to list

    d = cohens_d(a, b)
    assert d < 0  # a is much smaller than b

def test_weight_analysis():
    model = DummyModel()

    norms = get_weight_norms(model)
    assert 'linear1.weight' in norms
    assert 'l2' in norms['linear1.weight']

    ranks = compute_effective_rank(model)
    assert 'linear1.weight' in ranks
    assert ranks['linear1.weight'] > 0

    sparsity = measure_weight_sparsity(model, threshold=10.0)
    assert 'linear1.weight' in sparsity
    assert np.isclose(sparsity['linear1.weight'], 1.0) # all initial weights are < 10

    # Test gradients
    x = torch.randint(0, 10, (4, 5))
    y = torch.randint(0, 2, (4,))
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()

    grad_norms = track_gradient_norms(model)
    assert 'linear1.weight' in grad_norms

    # Test hessian (small iter for speed)
    loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    eig = compute_hessian_max_eigenvalue(model, loader, nn.CrossEntropyLoss(), num_iterations=2)
    assert isinstance(eig, float)

def test_interpretability():
    model = DummyModel()
    pure_model = DummyModel()
    collapsed_model = DummyModel()

    x = torch.randint(0, 10, (4, 5))
    y = torch.randint(0, 2, (4,))

    attrs = get_attention_head_attributions(model, x, y)
    assert 'linear1' in attrs

    patched_out = activation_patching(pure_model, collapsed_model, 'linear1', x)
    assert patched_out.shape == (4, 2)

    saliency = get_head_saliency_maps(model, x, y)
    assert saliency.shape == (2,)

    fourier = track_fourier_coefficients(model)
    assert fourier.shape == (10,)

def test_dashboard(tmp_path):
    levels = ['pure', 'low_collapse']
    sizes = ['small', 'large']

    loss_data = {
        'pure': {'steps': [1,2], 'loss': [1.0, 0.5], 'grokking_step': 1.5}
    }
    weight_data = {
        'pure': {'steps': [1,2], 'norms': [10.0, 8.0]}
    }
    entropy_data = {
        'pure': {'steps': [1,2], 'entropy': [2.0, 1.0]}
    }
    prob_mat = np.array([[1.0, 0.8], [0.1, 0.05]])
    fourier = {
        'pure': np.array([1.0, 0.5, 0.2])
    }

    out_file = tmp_path / "dash.pdf"
    generate_comparison_dashboard(
        loss_data, weight_data, entropy_data, prob_mat,
        levels, sizes, fourier, save_path=str(out_file)
    )
    assert out_file.exists()
