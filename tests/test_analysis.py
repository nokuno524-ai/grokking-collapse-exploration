import numpy as np
from src.analysis.grok_detector.stats import permutation_test_collapse

def test_permutation_test_collapse():
    pure = np.array([1.0, 1.0, 1.0, 0.9])
    collapse = np.array([0.1, 0.2, 0.1, 0.1])
    p_val = permutation_test_collapse(pure, collapse, n_permutations=100)
    assert p_val < 0.1 # Should be significant
