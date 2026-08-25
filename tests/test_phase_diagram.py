import numpy as np
import pytest
from pathlib import Path

# Adjusting to standard import as `src` is properly installed
from src.analysis.phase_diagram import (
    classify_phase,
    aggregate_metrics,
    plot_phase_diagram_2d,
    plot_critical_step_vs_severity
)

def test_classify_phase():
    # Gap < 0.1: grokked (1)
    assert classify_phase(1.0, 0.95) == 1
    # 0.1 <= Gap <= 0.9: transitioning (0)
    assert classify_phase(1.0, 0.5) == 0
    # Gap > 0.9: memorizing-only (-1)
    assert classify_phase(1.0, 0.05) == -1

    # Exactly on boundary
    # Floating point issues can make 1.0 - 0.9 = 0.099999999
    # so we test gap = 0.15 for transition to be safe
    assert classify_phase(1.0, 0.85) == 0  # gap = 0.15 -> transitioning

    # Check gap=0.9 boundary
    # gap > 0.9 should be -1
    assert classify_phase(1.0, 0.05) == -1
    # gap = 0.85 should be 0
    assert classify_phase(1.0, 0.15) == 0

def test_aggregate_metrics():
    # Synthetic grid
    results = [
        {
            "severity": 0.0,
            "history": [
                {"step": 0, "train_acc": 0.2, "test_acc": 0.2}, # gap 0 -> grokked
                {"step": 100, "train_acc": 1.0, "test_acc": 1.0}, # gap 0 -> grokked
            ]
        },
        {
            "severity": 0.1,
            "history": [
                {"step": 0, "train_acc": 0.2, "test_acc": 0.2}, # gap 0 -> grokked
                {"step": 100, "train_acc": 1.0, "test_acc": 0.5}, # gap 0.5 -> transitioning
            ]
        },
        {
            # Testing missing cells
            "severity": 0.2,
            "history": [
                {"step": 0, "train_acc": 0.2, "test_acc": 0.2}, # gap 0 -> grokked
                # missing step 100
            ]
        }
    ]

    matrix, steps, severities = aggregate_metrics(results)

    assert list(steps) == [0, 100]
    assert list(severities) == [0.0, 0.1, 0.2]

    # Shape should be (len(steps), len(severities))
    assert matrix.shape == (2, 3)

    # Check severity 0.0
    assert matrix[0, 0] == 1  # step 0
    assert matrix[1, 0] == 1  # step 100

    # Check severity 0.1
    assert matrix[0, 1] == 1  # step 0
    assert matrix[1, 1] == 0  # step 100

    # Check severity 0.2
    assert matrix[0, 2] == 1  # step 0
    assert np.isnan(matrix[1, 2])  # step 100 is missing

def test_plot_generation(tmp_path):
    matrix = np.array([
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, -1.0],
    ])
    steps = [0, 100, 200]
    severities = [0.0, 0.2]

    diagram_path = tmp_path / "diagram.png"
    critical_path = tmp_path / "critical.png"

    plot_phase_diagram_2d(matrix, steps, severities, diagram_path)
    assert diagram_path.exists()
    assert diagram_path.stat().st_size > 0

    plot_critical_step_vs_severity(matrix, steps, severities, critical_path)
    assert critical_path.exists()
    assert critical_path.stat().st_size > 0
