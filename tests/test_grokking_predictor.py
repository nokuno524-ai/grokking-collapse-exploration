import pytest
import numpy as np
from src.grokking_predictor import (
    extract_early_features,
    train_grokking_predictor,
    predict_grokking
)

def test_extract_early_features():
    history = [
        {"train_loss": 2.0, "test_loss": 2.1, "weight_norm": 10.0, "grad_noise_scale": 1.0, "attention_entropy": 2.0},
        {"train_loss": 1.0, "test_loss": 1.5, "weight_norm": 15.0, "grad_noise_scale": 0.5, "attention_entropy": 1.5},
        {"train_loss": 0.5, "test_loss": 1.2, "weight_norm": 20.0, "grad_noise_scale": 0.1, "attention_entropy": 1.0},
    ]

    features = extract_early_features(history, n_steps=3)

    assert "final_loss_gap" in features
    assert "mean_loss_gap" in features
    assert "weight_norm_slope" in features
    assert "mean_grad_noise" in features
    assert "mean_attn_entropy" in features

    # Final gap: 1.2 - 0.5 = 0.7
    assert np.isclose(features["final_loss_gap"], 0.7)
    # Norms: 10, 15, 20 => slope is 5.0
    assert np.isclose(features["weight_norm_slope"], 5.0)

def test_train_grokking_predictor():
    # Create dummy data:
    # Positive examples (grokking): high slope, low loss gap
    # Negative examples (no grokking): low slope, high loss gap
    histories = []
    labels = []

    for i in range(10):
        # Grokking
        h_pos = [
            {"train_loss": 1.0, "test_loss": 1.1, "weight_norm": 10.0 + i, "grad_noise_scale": 1.0, "attention_entropy": 2.0},
            {"train_loss": 0.5, "test_loss": 0.6, "weight_norm": 15.0 + i, "grad_noise_scale": 1.0, "attention_entropy": 2.0},
        ]
        histories.append(h_pos)
        labels.append(1)

        # No grokking
        h_neg = [
            {"train_loss": 1.0, "test_loss": 2.0, "weight_norm": 10.0, "grad_noise_scale": 1.0, "attention_entropy": 2.0},
            {"train_loss": 0.1, "test_loss": 3.0, "weight_norm": 10.1, "grad_noise_scale": 1.0, "attention_entropy": 2.0},
        ]
        histories.append(h_neg)
        labels.append(0)

    model, importances = train_grokking_predictor(histories, labels, n_steps=2)

    assert len(importances) == 5

    # Test prediction
    test_h = [
        {"train_loss": 1.0, "test_loss": 1.1, "weight_norm": 10.0, "grad_noise_scale": 1.0, "attention_entropy": 2.0},
        {"train_loss": 0.5, "test_loss": 0.6, "weight_norm": 15.0, "grad_noise_scale": 1.0, "attention_entropy": 2.0},
    ]
    pred = predict_grokking(model, test_h, n_steps=2)
    assert pred == 1
