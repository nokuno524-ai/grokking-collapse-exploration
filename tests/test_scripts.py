import pytest
from scripts.run_scaling import get_model_sizes, get_data_sizes, get_collapse_severities

def test_get_model_sizes():
    sizes = get_model_sizes()
    assert len(sizes) == 3
    assert sizes[0]["name"] == "tiny"
    assert sizes[1]["name"] == "small"
    assert sizes[2]["name"] == "base"

def test_get_data_sizes():
    sizes = get_data_sizes()
    assert len(sizes) == 4
    assert 0.2 in sizes

def test_get_collapse_severities():
    sevs = get_collapse_severities()
    assert "pure" in sevs
    assert "medium_collapse" in sevs
