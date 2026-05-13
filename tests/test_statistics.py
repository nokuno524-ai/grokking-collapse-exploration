import pytest
import numpy as np
from src.analysis.statistics import (
    compute_statistics,
    compare_conditions,
    anova_across_conditions,
    bootstrap_ci
)

def test_compute_statistics():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = compute_statistics(data)

    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0
    assert np.isclose(stats["std"], 1.5811388300841898) # sample std
    assert stats["ci_lower"] < 3.0
    assert stats["ci_upper"] > 3.0

def test_compare_conditions():
    # Two identical distributions
    cond_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    cond_b = [1.0, 2.0, 3.0, 4.0, 5.0]
    res_ident = compare_conditions(cond_a, cond_b)

    assert res_ident["t_stat"] == 0.0
    assert res_ident["p_value"] == 1.0
    assert res_ident["cohens_d"] == 0.0

    # Two very different distributions
    cond_c = [10.0, 11.0, 12.0, 13.0, 14.0]
    res_diff = compare_conditions(cond_a, cond_c)

    assert res_diff["t_stat"] < 0
    assert res_diff["p_value"] < 0.05
    assert res_diff["cohens_d"] < 0

def test_anova_across_conditions():
    cond_a = [1.0, 2.0, 3.0]
    cond_b = [1.1, 2.1, 3.1]
    cond_c = [10.0, 11.0, 12.0]

    res = anova_across_conditions(cond_a, cond_b, cond_c)
    assert res["p_value"] < 0.05
    assert res["f_stat"] > 0

def test_bootstrap_ci():
    np.random.seed(42)
    # Generate 100 points from normal with mean 10 and std 2
    data = np.random.normal(10, 2, 100).tolist()

    ci_lower, ci_upper = bootstrap_ci(data, num_resamples=1000)
    assert ci_lower < 10.5
    assert ci_upper > 9.5
    assert ci_lower < ci_upper
