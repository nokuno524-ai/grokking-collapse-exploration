import pytest
import torch
import numpy as np
from src.data import generate_modular_arithmetic, DatasetConfig
from scripts.detect_synthetic import extract_features, evaluate_detection
from src.model import ModularArithmeticTransformer

def test_data_generation_mask():
    """Test that the collapse mask is properly stored and matches expected counts."""
    config = DatasetConfig(prime=59, collapse_level=0.1, collapse_severity=0.5, seed=42)
    train_in, train_tgt, test_in, test_tgt, mask = generate_modular_arithmetic(config, return_mask=True)


    assert mask.shape[0] == train_in.shape[0]
    assert mask.dtype == torch.bool

    # Check that about 10% of training data is flagged as synthetic
    expected_synthetic = int(len(train_in) * 0.1)
    actual_synthetic = mask.sum().item()
    assert actual_synthetic == expected_synthetic

def test_extract_features():
    """Test that feature extraction produces correct shapes and types."""
    model = ModularArithmeticTransformer(prime=11, d_model=16, n_heads=2, d_ff=32)

    batch_size = 4
    inputs = torch.randint(0, 11, (batch_size, 2))
    targets = torch.randint(0, 11, (batch_size,))

    probe_features, loss = extract_features(model, inputs, targets)

    # Feature size should be d_model (h) + d_model (target_embed) = 32
    assert probe_features.shape == (batch_size, 32)
    assert loss.shape == (batch_size,)
    assert isinstance(probe_features, np.ndarray)
    assert isinstance(loss, np.ndarray)

def test_evaluate_detection_edge_cases():
    """Test evaluation logic gracefully handles edge cases like only 1 class."""
    probe_features = np.random.randn(10, 32)
    loss_features = np.random.randn(10)
    target_freq = np.random.randn(10)

    # Edge case: No synthetic data (1 class)
    is_synthetic = np.zeros(10, dtype=bool)
    metrics = evaluate_detection(probe_features, loss_features, target_freq, is_synthetic)
    assert metrics["probe_auroc"] == 0.5
    assert metrics["probe_ap"] == 0.0

    # Normal case: mix of 0s and 1s, enough to cover CV=5
    is_synthetic = np.array([True]*10 + [False]*20)
    probe_features = np.random.randn(30, 32)
    loss_features = np.random.randn(30)
    target_freq = np.random.randn(30)
    metrics = evaluate_detection(probe_features, loss_features, target_freq, is_synthetic)
    # the nan check can still happen if np.mean fails, so check if np.isnan(metrics['probe_auroc']) is False
    assert not np.isnan(metrics["probe_auroc"])
    assert 0.0 <= metrics["probe_auroc"] <= 1.0
    assert 0.0 <= metrics["loss_auroc"] <= 1.0
    assert 0.0 <= metrics["freq_auroc"] <= 1.0
