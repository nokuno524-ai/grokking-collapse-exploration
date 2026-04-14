import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import metrics

def test_mode_collapse_score():
    # Uniform
    predictions = torch.arange(59).repeat(10)
    score = metrics.mode_collapse_score(predictions, 59)
    assert score < 0.01

    # Complete collapse
    predictions = torch.zeros(590, dtype=torch.long)
    score = metrics.mode_collapse_score(predictions, 59)
    assert score > 0.99

def test_kl_divergence_shift():
    # Same
    predictions = torch.arange(59).repeat(10)
    targets = torch.arange(59).repeat(10)
    kl = metrics.kl_divergence_shift(predictions, targets, 59)
    assert kl < 0.01

    # Different
    predictions = torch.zeros(590, dtype=torch.long)
    targets = torch.arange(59).repeat(10)
    kl = metrics.kl_divergence_shift(predictions, targets, 59)
    assert kl > 1.0

def test_memorization_score():
    assert metrics.memorization_score(1.0, 1.0) == 0.0
    assert metrics.memorization_score(1.0, 0.0) == 1.0
    assert metrics.memorization_score(0.5, 0.5) == 0.0
