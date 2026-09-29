import pytest
from src.rigorous_stats.cliff_stability import find_cliff_location, analyze_cliff_stability

def test_find_cliff_location():
    history = [
        {"step": 10, "test_acc": 0.5},
        {"step": 20, "test_acc": 0.8},
        {"step": 30, "test_acc": 0.96},
        {"step": 40, "test_acc": 0.98},
    ]
    assert find_cliff_location(history, threshold=0.95) == 30

def test_find_cliff_location_never():
    history = [
        {"step": 10, "test_acc": 0.5},
        {"step": 20, "test_acc": 0.8},
    ]
    assert find_cliff_location(history, threshold=0.95) is None

def test_find_cliff_location_empty():
    assert find_cliff_location([], threshold=0.95) is None

def test_analyze_cliff_stability_all_identical():
    runs_data = [
        {"grokking_step": 50},
        {"grokking_step": 50},
        {"grokking_step": 50},
    ]
    res = analyze_cliff_stability(runs_data)
    assert res["num_runs"] == 3
    assert res["num_grokked"] == 3
    assert res["mean_cliff"] == 50.0
    assert res["std_cliff"] == 0.0

def test_analyze_cliff_stability_one_diverged():
    runs_data = [
        {"grokking_step": 50},
        {"grokking_step": 50},
        {"grokking_step": None}, # Diverged / never grokked
    ]
    res = analyze_cliff_stability(runs_data)
    assert res["num_runs"] == 3
    assert res["num_grokked"] == 2
    assert res["grokking_prob"] == 2/3
    assert res["mean_cliff"] == 50.0

def test_analyze_cliff_stability_none_grokked():
    runs_data = [
        {"grokking_step": None},
        {"grokking_step": None},
    ]
    res = analyze_cliff_stability(runs_data)
    assert res["num_runs"] == 2
    assert res["num_grokked"] == 0
    assert "mean_cliff" not in res
