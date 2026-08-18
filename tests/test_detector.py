import pytest
from src.analysis.results_analysis import detect_grokking_step

def test_detect_grokking_step():
    # Never reaches
    assert detect_grokking_step([0.1, 0.2, 0.3]) is None

    # Reaches but drops immediately
    assert detect_grokking_step([0.1, 0.96, 0.8, 0.9, 0.9], window_size=3) is None

    # Stable for window
    assert detect_grokking_step([0.1, 0.96, 0.97, 0.98, 0.9], window_size=3) == 1

    # Exact threshold
    assert detect_grokking_step([0.1, 0.95, 0.95, 0.95, 0.8], window_size=3) == 1

    # Empty
    assert detect_grokking_step([]) is None

    # Too short
    assert detect_grokking_step([0.99, 0.99], window_size=3) is None
