import numpy as np
from typing import Tuple, List, Callable, Union

def cohens_d(group1: Union[np.ndarray, List[float]], group2: Union[np.ndarray, List[float]], paired: bool = False) -> float:
    """
    Compute Cohen's d for two groups of data.
    If paired=True, computes Cohen's d for paired samples (mean of diffs / std of diffs).
    Otherwise computes standard independent Cohen's d.
    Handles zero variance and n=1 edge cases.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    if len(g1) == 0 or len(g2) == 0:
        return 0.0

    if paired:
        if len(g1) != len(g2):
            raise ValueError("Groups must be the same size for paired Cohen's d.")
        if len(g1) == 1:
            return 0.0 # Cannot compute variance with n=1

        diffs = g2 - g1
        variance = np.var(diffs, ddof=1)
        if variance == 0:
            return 0.0
        return float(np.mean(diffs) / np.sqrt(variance))
    else:
        n1, n2 = len(g1), len(g2)
        if n1 == 1 and n2 == 1:
            return 0.0 # Cannot compute variance with n=1

        var1 = np.var(g1, ddof=1) if n1 > 1 else 0.0
        var2 = np.var(g2, ddof=1) if n2 > 1 else 0.0

        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2) if (n1 + n2 > 2) else 0.0

        if pooled_var == 0:
            return 0.0

        return float((np.mean(g2) - np.mean(g1)) / np.sqrt(pooled_var))

def bootstrap_ci(data: Union[np.ndarray, List[float]], stat_func: Callable = np.mean, n_boot: int = 1000, ci: float = 95.0, seed: int = 42) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a 1D array.
    """
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return (0.0, 0.0)
    if len(data) == 1:
        val = float(stat_func(data))
        return (val, val)

    rng = np.random.default_rng(seed)

    # Sample with replacement
    boot_samples = rng.choice(data, size=(n_boot, len(data)), replace=True)
    boot_stats = np.apply_along_axis(stat_func, 1, boot_samples)

    lower_bound = np.percentile(boot_stats, (100 - ci) / 2.0)
    upper_bound = np.percentile(boot_stats, 100 - (100 - ci) / 2.0)

    return float(lower_bound), float(upper_bound)

def check_replication(effect_sizes: List[float]) -> bool:
    """
    Checks whether the effect replicates across seeds.
    A simple check: do all non-zero effect sizes have the same sign?
    (And is there at least one non-zero effect?)
    """
    if not effect_sizes:
        return False

    signs = [np.sign(e) for e in effect_sizes if abs(e) > 1e-6]

    if not signs:
        return False # All effects are zero

    first_sign = signs[0]
    return all(s == first_sign for s in signs)
