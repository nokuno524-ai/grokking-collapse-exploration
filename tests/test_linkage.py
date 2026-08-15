import json
import pytest
import numpy as np
from pathlib import Path
from analysis import linkage

@pytest.fixture
def dummy_run_data():
    return {
        "config": {"condition_name": "test_condition", "collapse_level": 0.15},
        "history": [
            {"step": 100, "test_acc": 0.1, "weight_norm": 10.0, "attention_entropy": 2.5},
            {"step": 200, "test_acc": 0.5, "weight_norm": 20.0, "attention_entropy": 2.0},
            {"step": 300, "test_acc": 0.995, "weight_norm": 15.0, "attention_entropy": 1.5},
            {"step": 400, "test_acc": 1.0, "weight_norm": 15.0, "attention_entropy": 1.0}
        ]
    }

def test_extract_metrics(dummy_run_data):
    metrics = linkage.extract_metrics(dummy_run_data, tau=0.99)
    assert metrics['grok_step'] == 300
    assert metrics['grok_success'] is True
    assert metrics['final_test_acc'] == 1.0
    assert metrics['wn_drop_pct'] == (20.0 - 15.0) / 20.0
    assert metrics['condition_name'] == "test_condition"
    assert metrics['noise_level'] == 0.15

def test_extract_metrics_no_grok(dummy_run_data):
    # Alter history to not cross tau
    dummy_run_data["history"][2]["test_acc"] = 0.8
    dummy_run_data["history"][3]["test_acc"] = 0.85
    metrics = linkage.extract_metrics(dummy_run_data, tau=0.99)
    assert metrics['grok_step'] is None
    assert metrics['grok_success'] is False
    assert metrics['final_test_acc'] == 0.85

def test_fit_delay_vs_severity():
    severities = [0.1, 0.2, 0.3, 0.4]
    delays = [1000, 1500, None, 2200]
    popt, stderr = linkage.fit_delay_vs_severity(severities, delays)
    assert popt is not None
    assert len(popt) == 2
    assert len(stderr) == 2

def test_fit_success_vs_severity():
    severities = [0.1, 0.2, 0.3, 0.4]
    successes = [1, 1, 0, 0]
    popt, stderr = linkage.fit_success_vs_severity(severities, successes)
    assert popt is not None
    assert len(popt) == 3
