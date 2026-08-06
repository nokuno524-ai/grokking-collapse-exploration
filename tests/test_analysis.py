import numpy as np
import pytest
from src.analysis.information_flow import compute_mutual_information, information_bottleneck_curve
from src.analysis.phase_detection import detect_grokking_point, detect_collapse_onset, grokking_gap_metrics

def test_compute_mutual_information():
    # Identical arrays should have high MI
    x = np.array([0, 1, 0, 1, 0, 1])
    y = np.array([0, 1, 0, 1, 0, 1])
    mi = compute_mutual_information(x, y, num_bins=2)
    assert mi > 0.5  # Should be close to 1 bit

    # Random arrays should have low MI
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randint(0, 2, 100)
    mi2 = compute_mutual_information(x, y, num_bins=10)
    assert mi2 < 0.5

def test_information_bottleneck_curve():
    epochs_acts = [
        np.random.randn(50, 5),
        np.random.randn(50, 5),
        np.random.randn(50, 5)
    ]
    inputs = np.random.randint(0, 5, (50, 2))
    labels = np.random.randint(0, 5, 50)

    mi_xt, mi_ty = information_bottleneck_curve(epochs_acts, inputs, labels, num_bins=5)
    assert len(mi_xt) == 3
    assert len(mi_ty) == 3

def test_detect_grokking_point():
    train_loss = [2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.01]
    val_loss =   [2.0, 1.5, 1.2, 1.1, 1.0,  0.1,  0.05]

    idx = detect_grokking_point(train_loss, val_loss)
    # The drop happens from 1.0 -> 0.1 at index 5
    assert idx == 5

def test_detect_collapse_onset():
    wn = [10.0, 9.5, 9.0, 4.0, 3.5, 3.0]
    idx = detect_collapse_onset(wn, drop_threshold=0.2)
    # Drop from 9.0 to 4.0 occurs at index 3
    assert idx == 3

    wn_no_collapse = [10.0, 9.9, 9.8, 9.7]
    assert detect_collapse_onset(wn_no_collapse, drop_threshold=0.2) == -1

def test_grokking_gap_metrics():
    train_acc = [0.1, 0.5, 0.96, 0.98, 0.99, 1.0]
    val_acc   = [0.1, 0.2, 0.2,  0.3,  0.97, 0.99]

    gap = grokking_gap_metrics(train_acc, val_acc, memo_threshold=0.95, gen_threshold=0.95)
    # memo at idx 2, gen at idx 4 -> gap = 2
    assert gap == 2
