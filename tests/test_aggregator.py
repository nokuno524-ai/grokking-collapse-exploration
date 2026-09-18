import pytest
import numpy as np
from src.analysis.aggregator import compute_ci, aggregate_and_report

def test_compute_ci():
    # Simple test for mean and CI
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    m, ci = compute_ci(data)
    assert m == 3.0
    assert ci > 0.0

    # Test edge case with 1 element
    m, ci = compute_ci([1.0])
    assert m == 1.0
    assert ci == 0.0

def test_aggregator_empty_dir(tmp_path):
    df = aggregate_and_report(str(tmp_path), ["condA"], metric="acc")
    assert len(df) == 1
    assert np.isnan(df.iloc[0]["mean"])
