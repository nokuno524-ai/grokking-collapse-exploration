import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from src.analysis.grok_detector.detectors import threshold_detector, binary_segmentation_detector, bootstrap_ci, detect_cliffs
from src.analysis.grok_detector.stats import holm_correction
from src.analysis.grok_detector.run_aggregator import parse_log_file, MalformedRunError

def test_threshold_detector():
    steps = np.arange(0, 1000, 10)
    # Synthetic accs: 0.1 until step 500, then 1.0
    accs = np.array([0.1 if s < 500 else 1.0 for s in steps])

    cliff = threshold_detector(steps, accs, threshold=0.9, dwell_k=5)
    assert cliff == 500

def test_threshold_detector_censored():
    steps = np.arange(0, 1000, 10)
    # Never crosses 0.9
    accs = np.array([0.1 for _ in steps])

    cliff = threshold_detector(steps, accs, threshold=0.9, dwell_k=5)
    assert cliff is None

def test_binary_segmentation_detector():
    steps = np.arange(0, 1000, 10)
    accs = np.array([0.1 if s < 500 else 1.0 for s in steps])

    cliff = binary_segmentation_detector(steps, accs)
    # Binary segmentation should find the exact split point
    assert cliff == 500

def test_binary_segmentation_censored():
    steps = np.arange(0, 1000, 10)
    # Noise, no cliff
    accs = np.random.uniform(0.1, 0.2, len(steps))

    cliff = binary_segmentation_detector(steps, accs)
    assert cliff is None

def test_bootstrap_ci():
    # Synthetic normal data, true median ~ 50
    data = np.random.normal(50, 5, 1000)
    median, lower, upper = bootstrap_ci(data, num_bootstrap=500, ci_level=0.95)

    assert 48 < median < 52
    assert lower < median < upper

def test_holm_correction():
    p_values = [0.01, 0.04, 0.03, 0.005]
    adj_p = holm_correction(p_values)

    # smallest p-val: 0.005 * 4 = 0.02
    # next: 0.01 * 3 = 0.03
    # next: 0.03 * 2 = 0.06
    # largest: 0.04 * 1 = 0.04 -> bumped to 0.06 due to monotonicity

    assert adj_p[3] == 0.02
    assert adj_p[0] == 0.03
    assert adj_p[2] == 0.06
    assert adj_p[1] == 0.06

def test_parse_log_valid(tmp_path):
    log_data = {
        "config": {"noise_fraction": 0.1},
        "history": [
            {"step": 0, "test_acc": 0.1},
            {"step": 10, "test_acc": 0.9}
        ]
    }
    file_path = tmp_path / "results.json"
    with open(file_path, "w") as f:
        json.dump(log_data, f)

    parsed = parse_log_file(file_path)
    assert parsed["config"]["noise_fraction"] == 0.1
    assert len(parsed["history"]) == 2

def test_parse_log_malformed_missing_config(tmp_path):
    log_data = {
        "history": [
            {"step": 0, "test_acc": 0.1},
        ]
    }
    file_path = tmp_path / "results.json"
    with open(file_path, "w") as f:
        json.dump(log_data, f)

    with pytest.raises(MalformedRunError):
        parse_log_file(file_path)
