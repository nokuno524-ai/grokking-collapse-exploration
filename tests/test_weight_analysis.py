import torch
import torch.nn as nn
import numpy as np
import pytest
from src.weight_analysis import WeightAnalyzer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)

def test_compute_weight_norms():
    model = SimpleModel()
    analyzer = WeightAnalyzer()
    norms = analyzer.compute_weight_norms(model)

    assert "fc1.weight" in norms
    assert "fc2.weight" in norms
    assert isinstance(norms["fc1.weight"], float)
    assert norms["fc1.weight"] > 0

def test_compute_effective_rank():
    # Rank 2 matrix
    matrix = torch.zeros((10, 10))
    matrix[0, 0] = 10.0
    matrix[1, 1] = 5.0

    analyzer = WeightAnalyzer()
    rank = analyzer.compute_effective_rank(matrix, threshold=0.99)
    assert rank == 2

def test_compute_singular_spectrum():
    matrix = torch.zeros((5, 5))
    matrix[0, 0] = 4.0
    matrix[1, 1] = 2.0
    matrix[2, 2] = 1.0

    analyzer = WeightAnalyzer()
    spectrum = analyzer.compute_singular_spectrum(matrix)

    assert spectrum.shape == (5,)
    assert spectrum[0] == 1.0 # Normalized
    assert spectrum[1] == 0.5
    assert spectrum[2] == 0.25

def test_compare_weight_distributions():
    model_a = SimpleModel()
    model_b = SimpleModel()

    analyzer = WeightAnalyzer()
    distances = analyzer.compare_weight_distributions(model_a, model_b)

    assert "fc1.weight" in distances
    assert "fc2.weight" in distances
    assert distances["fc1.weight"] >= 0
