import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats

def compute_median_iqr(values: List[float]) -> Tuple[float, float, float]:
    """
    Computes median and Interquartile Range (IQR).

    Returns:
        (median, q25, q75)
    """
    if not values:
        return np.nan, np.nan, np.nan
    arr = np.array(values)
    median = np.median(arr)
    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    return float(median), float(q25), float(q75)


def holm_mann_whitney(groups: Dict[str, List[float]], control_group: str) -> Dict[str, Dict[str, Any]]:
    """
    Performs pairwise Mann-Whitney U tests against a control group,
    with Holm's step-down multiple comparison correction.

    Args:
        groups: Dictionary mapping group names to lists of values.
        control_group: The name of the group to compare against.

    Returns:
        A dictionary mapping compared group names to their test statistics
        and corrected p-values.
    """
    if control_group not in groups:
        raise ValueError(f"Control group '{control_group}' not found in groups.")

    control_vals = groups[control_group]

    results = []
    for name, vals in groups.items():
        if name == control_group:
            continue

        # If one group is empty or all same values, test might complain or give p=1.0
        if len(control_vals) == 0 or len(vals) == 0:
            stat, p_val = np.nan, np.nan
        else:
            try:
                # alternative='two-sided' is default
                stat, p_val = stats.mannwhitneyu(vals, control_vals)
            except ValueError:
                stat, p_val = np.nan, np.nan

        results.append({
            'group': name,
            'stat': stat,
            'p_val': p_val
        })

    # Filter valid p-values for correction
    valid_results = [r for r in results if not np.isnan(r['p_val'])]

    # Sort by p-value (ascending)
    valid_results.sort(key=lambda x: x['p_val'])

    m = len(valid_results)

    # Holm's step-down procedure
    corrected_results = {}
    for i, r in enumerate(valid_results):
        # Step-down correction: p_corr = p * (m - i)
        # Note: True Holm correction takes max(p_corr_{i-1}, p * (m - i)) to ensure monotonicity,
        # bounded by 1.0.
        multiplier = m - i
        raw_p = r['p_val']
        adj_p = min(1.0, raw_p * multiplier)

        if i > 0:
            prev_adj_p = corrected_results[valid_results[i-1]['group']]['p_adj']
            adj_p = max(prev_adj_p, adj_p)

        corrected_results[r['group']] = {
            'stat': r['stat'],
            'p_raw': raw_p,
            'p_adj': adj_p
        }

    # Add NaN results back
    for r in results:
        if np.isnan(r['p_val']):
            corrected_results[r['group']] = {
                'stat': np.nan,
                'p_raw': np.nan,
                'p_adj': np.nan
            }

    return corrected_results


def permutation_spearman(
    x: List[float],
    y: List[float],
    num_permutations: int = 10000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Computes Spearman rank correlation and a permutation p-value.

    Args:
        x: First variable.
        y: Second variable.
        num_permutations: Number of permutations for the null distribution.
        seed: Random seed for reproducibility.

    Returns:
        (correlation, permutation_p_value)
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if len(x) < 2:
        return np.nan, np.nan

    x_arr = np.array(x)
    y_arr = np.array(y)

    # Calculate observed Spearman correlation
    res = stats.spearmanr(x_arr, y_arr)
    obs_corr = res.correlation

    # Handle exact 1.0 or NaN cases
    if np.isnan(obs_corr):
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    # Permutation test
    # We permute y, calculate correlation with x, and count how many times
    # the absolute permuted correlation is >= absolute observed correlation
    abs_obs = abs(obs_corr)
    count = 0

    y_perm = y_arr.copy()
    for _ in range(num_permutations):
        rng.shuffle(y_perm)
        perm_corr = stats.spearmanr(x_arr, y_perm).correlation
        if abs(perm_corr) >= abs_obs:
            count += 1

    p_val = count / num_permutations

    return obs_corr, p_val
