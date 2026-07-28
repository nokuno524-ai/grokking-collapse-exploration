import pytest
from src.experiments.run_scaling import get_scaling_matrix

def test_scaling_matrix_generation():
    matrix = get_scaling_matrix()

    assert "d_models" in matrix
    assert "n_heads" in matrix
    assert "train_fractions" in matrix
    assert "collapse_levels" in matrix

    assert len(matrix["d_models"]) > 0
    assert len(matrix["n_heads"]) > 0
    assert len(matrix["train_fractions"]) > 0
    assert len(matrix["collapse_levels"]) > 0
