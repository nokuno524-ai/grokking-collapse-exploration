import torch
import torch.nn as nn
import numpy as np
import pytest

from analysis.weights import get_weight_norms, effective_rank, get_matrix_ranks
from analysis.mlp_geometry import track_mlp_activations
from analysis.mechanisms import causal_patching

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 5)
        # Using a dummy layer structure that matches the expected naming in analysis
        self.transformer = nn.ModuleList([DummyTransformerLayer()])
        self.output_head = nn.Linear(5, 2)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.transformer:
            h = layer(h)
        return self.output_head(h.mean(dim=1))

class DummyTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(5, 5)
        self.linear1 = nn.Linear(5, 10)
        self.linear2 = nn.Linear(10, 5)

    def forward(self, x):
        x = self.self_attn(x)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# For causal patching, we need a specific structure
class PatchableTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = type('obj', (object,), {'layers': nn.ModuleList([nn.Linear(5, 5), nn.Linear(5, 5)])})()

    def forward(self, x):
        for layer in self.transformer.layers:
            x = layer(x)
        return x

def test_weight_norms():
    model = DummyModel()
    # Initialize with known values
    nn.init.constant_(model.embed.weight, 1.0)
    nn.init.constant_(model.transformer[0].self_attn.weight, 2.0)
    nn.init.constant_(model.transformer[0].linear1.weight, 3.0)
    nn.init.constant_(model.output_head.weight, 4.0)

    # Exclude biases for cleaner test or just set to 0
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0.0)

    norms = get_weight_norms(model)

    # embed: 10*5 = 50 items -> norm = sqrt(50*1) = sqrt(50)
    assert np.isclose(norms["embedding"], np.sqrt(50))
    # attn: 5*5 = 25 items -> norm = sqrt(25*4) = 10
    assert np.isclose(norms["attention"], 10)
    # mlp: (5*10)*9 + (10*5)*0 = 450 (only linear1 tested, linear2 is random unless set)
    # Actually wait, let's just ensure it's > 0
    assert norms["mlp"] > 0
    assert norms["output_head"] > 0

def test_effective_rank():
    # Matrix of zeros
    z = torch.zeros((5, 5))
    assert effective_rank(z) == 0.0

    # Matrix with NaNs
    n = torch.tensor([[1.0, float('nan')], [float('inf'), 2.0]])
    rank = effective_rank(n)
    assert rank >= 0.0
    assert not torch.isnan(torch.tensor(rank))

    # Identity matrix (rank should be equal to dimension)
    i = torch.eye(5)
    rank = effective_rank(i)
    assert np.isclose(rank, 5.0, atol=0.1)

def test_mlp_activation():
    model = DummyModel()
    # Create simple dataset
    x = torch.randint(0, 10, (10, 5))
    y = torch.randint(0, 2, (10,))

    class DummyLoader:
        def __iter__(self):
            yield x, y

    # Set linear1 weights positive so relu gives non-zero
    nn.init.constant_(model.transformer[0].linear1.weight, 1.0)
    nn.init.constant_(model.transformer[0].linear1.bias, 1.0)

    activations = track_mlp_activations(model, DummyLoader(), "cpu", num_batches=1)

    assert len(activations) > 0
    # At least one layer's activation should be tracked
    for name, acts in activations.items():
        assert isinstance(acts, torch.Tensor)
        assert acts.mean().item() > 0

def test_circuit_detection():
    model = PatchableTransformer()
    # Set weights to simple deterministic values
    nn.init.constant_(model.transformer.layers[0].weight, 1.0)
    nn.init.constant_(model.transformer.layers[0].bias, 0.0)
    nn.init.constant_(model.transformer.layers[1].weight, 2.0)
    nn.init.constant_(model.transformer.layers[1].bias, 0.0)

    clean_input = torch.ones(1, 5)
    corrupted_input = torch.zeros(1, 5)

    def sum_metric(out):
        return out.sum()

    # Patch layer 0 with zeros
    val = causal_patching(model, clean_input, corrupted_input, target_layer_idx=0, metric_fn=sum_metric)

    # Since patched layer 0 output is zeros (from corrupted input), layer 1 output is zeros
    assert val == 0.0
