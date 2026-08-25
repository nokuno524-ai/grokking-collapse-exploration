import numpy as np
import pytest
from pathlib import Path
from src.analysis.phase_diagram import (
    classify_phase,
    aggregate_metrics,
    build_phase_matrix,
    find_critical_steps,
    plot_phase_diagram,
    plot_critical_step
)

def test_classify_phase():
    # Grokked
    assert classify_phase(1.0, 0.96) == 2
    # Memorizing
    assert classify_phase(1.0, 0.5) == 0
    # Transitioning / not memorized yet
    assert classify_phase(0.8, 0.2) == 1

def test_aggregate_metrics():
    data = {
        0.3: {
            100: [(1.0, 0.5), (0.8, 0.3)],
            200: [(1.0, 1.0)]
        }
    }
    agg = aggregate_metrics(data)
    assert 0.3 in agg
    assert 100 in agg[0.3]
    assert np.isclose(agg[0.3][100][0], 0.9)
    assert np.isclose(agg[0.3][100][1], 0.4)
    assert np.isclose(agg[0.3][200][0], 1.0)
    assert np.isclose(agg[0.3][200][1], 1.0)

def test_build_phase_matrix():
    agg_data = {
        0.3: {100: (1.0, 0.5), 200: (1.0, 1.0)}, # step 100: Mem, step 200: Grok
        0.5: {100: (0.8, 0.2)}                   # step 100: Trans, step 200: Missing
    }
    severities = [0.3, 0.5, 0.7]
    steps = [100, 200]
    matrix = build_phase_matrix(agg_data, severities, steps)

    assert matrix.shape == (2, 3)

    # severity 0.3 (index 0)
    assert matrix[0, 0] == 0  # step 100
    assert matrix[1, 0] == 2  # step 200

    # severity 0.5 (index 1)
    assert matrix[0, 1] == 1  # step 100
    assert np.isnan(matrix[1, 1])  # step 200 missing

    # severity 0.7 (index 2)
    assert np.isnan(matrix[0, 2])
    assert np.isnan(matrix[1, 2])

def test_find_critical_steps():
    agg_data = {
        0.3: {100: (1.0, 0.5), 200: (1.0, 1.0)},
        0.5: {100: (1.0, 0.2), 200: (1.0, 0.3)},
        0.7: {}
    }
    severities = [0.3, 0.5, 0.7]
    steps = [100, 200]
    critical_steps = find_critical_steps(agg_data, severities, steps)

    assert critical_steps[0] == 200.0
    assert np.isnan(critical_steps[1])
    assert np.isnan(critical_steps[2])

def test_plot_generation(tmp_path):
    matrix = np.array([
        [0, 1, np.nan],
        [2, 0, np.nan]
    ])
    severities = [0.1, 0.2, 0.3]
    steps = [100, 200]

    phase_path = tmp_path / "phase.png"
    plot_phase_diagram(matrix, severities, steps, phase_path)
    assert phase_path.exists()
    assert phase_path.stat().st_size > 0

    c_steps = [100.0, np.nan, np.nan]
    crit_path = tmp_path / "crit.png"
    plot_critical_step(severities, c_steps, crit_path)
    assert crit_path.exists()
    assert crit_path.stat().st_size > 0
