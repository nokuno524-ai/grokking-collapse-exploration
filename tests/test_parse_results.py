"""Tests for the parse_results analysis script."""

import json
import os
import pytest
import tempfile
from analysis.parse_results import (
    parse_training_log,
    compute_summary,
    compare_conditions,
)

@pytest.fixture
def sample_json_log(tmp_path):
    """Fixture providing a temporary JSON log file."""
    data = {
        "history": [
            {"step": 100, "train_loss": 3.5, "test_loss": 4.5, "train_acc": 0.1, "test_acc": 0.01},
            {"step": 200, "train_loss": 2.0, "test_loss": 3.0, "train_acc": 0.5, "test_acc": 0.4},
            {"step": 300, "train_loss": 0.5, "test_loss": 1.0, "train_acc": 0.9, "test_acc": 0.96},
            {"step": 400, "train_loss": 0.1, "test_loss": 0.5, "train_acc": 1.0, "test_acc": 0.99},
        ]
    }
    filepath = tmp_path / "results.json"
    with open(filepath, "w") as f:
        json.dump(data, f)
    return str(filepath)

@pytest.fixture
def sample_text_log(tmp_path):
    """Fixture providing a temporary text log file."""
    log_content = (
        "Step   100 | train_loss=3.50 test_loss=4.50 | train_acc=0.10 test_acc=0.01 | time=1.0s\n"
        "Step   200 | train_loss=2.00 test_loss=3.00 | train_acc=0.50 test_acc=0.40 | time=1.0s\n"
        "Step   300 | train_loss=0.50 test_loss=1.00 | train_acc=0.90 test_acc=0.96 | time=1.0s\n"
        "Step   400 | train_loss=0.10 test_loss=0.50 | train_acc=1.00 test_acc=0.99 | time=1.0s\n"
    )
    filepath = tmp_path / "grok.out"
    with open(filepath, "w") as f:
        f.write(log_content)
    return str(filepath)

def test_parse_training_log_json(sample_json_log):
    """Test parsing a JSON log file."""
    data = parse_training_log(sample_json_log)
    assert 'history' in data
    assert len(data['history']) == 4
    assert data['history'][0]['step'] == 100
    assert data['history'][3]['test_acc'] == 0.99

def test_parse_training_log_text(sample_text_log):
    """Test parsing a text log file."""
    data = parse_training_log(sample_text_log)
    assert 'history' in data
    assert len(data['history']) == 4
    assert data['history'][0]['step'] == 100
    assert data['history'][3]['test_acc'] == 0.99
    assert data['history'][2]['test_loss'] == 1.00

def test_compute_summary():
    """Test computing summary statistics."""
    data = {
        'history': [
            {'step': 100, 'test_loss': 4.5, 'test_acc': 0.01},
            {'step': 200, 'test_loss': 3.0, 'test_acc': 0.40},
            {'step': 300, 'test_loss': 1.0, 'test_acc': 0.96},
            {'step': 400, 'test_loss': 0.5, 'test_acc': 0.99},
            {'step': 500, 'test_loss': 0.6, 'test_acc': 0.98},
        ]
    }
    summary = compute_summary(data)

    assert summary['min_loss'] == 0.5
    assert summary['max_accuracy'] == 0.99
    assert summary['step_of_best_accuracy'] == 400
    assert summary['convergence_step'] == 300

def test_compute_summary_no_convergence():
    """Test computing summary when model does not converge."""
    data = {
        'history': [
            {'step': 100, 'test_loss': 4.5, 'test_acc': 0.01},
            {'step': 200, 'test_loss': 4.0, 'test_acc': 0.10},
        ]
    }
    summary = compute_summary(data)

    assert summary['min_loss'] == 4.0
    assert summary['max_accuracy'] == 0.10
    assert summary['step_of_best_accuracy'] == 200
    assert summary['convergence_step'] is None

def test_compare_conditions():
    """Test generating comparison table string."""
    results = {
        'pure': {
            'min_loss': 0.5,
            'max_accuracy': 0.99,
            'step_of_best_accuracy': 400,
            'convergence_step': 300
        },
        'collapse': {
            'min_loss': 4.0,
            'max_accuracy': 0.10,
            'step_of_best_accuracy': 200,
            'convergence_step': None
        }
    }

    table = compare_conditions(results)

    assert "pure" in table
    assert "collapse" in table
    assert "0.5000" in table
    assert "4.0000" in table
    assert "N/A" in table
    assert "Condition" in table
    assert "Min Loss" in table
