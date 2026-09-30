import numpy as np
from typing import Tuple, Union
import math

def power_analysis(variance: float, target_diff: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """
    Computes the required sample size per group to detect a target difference.

    Uses the closed-form normal approximation for a two-sample t-test:
    n = 2 * (Z_alpha/2 + Z_beta)^2 * variance / (target_diff^2)

    Args:
        variance: The pooled variance (sigma squared) observed in pilot data.
        target_diff: The minimum meaningful effect size to detect (delta) between two means.
        alpha: Significance level (default 0.05).
        power: Desired statistical power (default 0.8).

    Returns:
        n_per_group: Required number of seeds (samples) per condition.
                     Returns NaN if inputs are invalid.
    """
    if variance <= 0 or target_diff == 0:
        return float("nan")

    try:
        from scipy.stats import norm
    except ImportError:
        # Fallback to hardcoded common values if scipy is unavailable
        # (Though in our environment it should be available)
        if np.isclose(alpha, 0.05) and np.isclose(power, 0.8):
            z_alpha = 1.96
            z_beta = 0.8416
        else:
            raise ImportError("scipy is required for non-standard alpha/power values.")
    else:
        # z-scores
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) ** 2) * variance / (target_diff ** 2)
    return n

def power_analysis_simulated(variance: float, target_diff: float, alpha: float = 0.05, power: float = 0.8, n_simulations: int = 1000) -> Union[int, float]:
    """
    Estimates required sample size per group via Monte Carlo simulation of t-tests.

    Args:
        variance: The pooled variance.
        target_diff: The target difference between means.
        alpha: Significance level.
        power: Desired power.
        n_simulations: Number of simulated t-tests per sample size.

    Returns:
        Required sample size per group, or NaN if inputs are invalid.
    """
    if variance <= 0 or target_diff == 0:
        return float("nan")

    try:
        from scipy.stats import ttest_ind
    except ImportError:
        raise ImportError("scipy is required for simulated power analysis.")

    std = np.sqrt(variance)
    rng = np.random.RandomState(42)  # For reproducibility in simulations

    # Search for n
    for n in range(2, 500):
        reject_count = 0
        for _ in range(n_simulations):
            group1 = rng.normal(0, std, n)
            group2 = rng.normal(target_diff, std, n)

            # If standard deviation is 0 (all samples identical), ttest_ind returns nan p-value
            if np.std(group1) == 0 and np.std(group2) == 0:
                pval = 1.0 if np.mean(group1) == np.mean(group2) else 0.0
            else:
                _, pval = ttest_ind(group1, group2, equal_var=True)

            if pval < alpha:
                reject_count += 1

        actual_power = reject_count / n_simulations
        if actual_power >= power:
            return n

    # If we hit the upper limit, just return the upper limit
    return 500
