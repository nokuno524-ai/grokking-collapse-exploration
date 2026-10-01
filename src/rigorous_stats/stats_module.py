

import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from scipy import stats


def bootstrap_ci_effect_size(x: np.ndarray, y: np.ndarray, effect_size_fn: callable, n_bootstraps: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes a bootstrap confidence interval for a given effect size function.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) < 2 or len(y) < 2:
        return (float('nan'), float('nan'))

    estimates = []
    for _ in range(n_bootstraps):
        x_boot = np.random.choice(x, size=len(x), replace=True)
        y_boot = np.random.choice(y, size=len(y), replace=True)
        estimates.append(effect_size_fn(x_boot, y_boot))

    estimates = np.array(estimates)
    estimates = estimates[~np.isnan(estimates)]

    if len(estimates) == 0:
        return (float('nan'), float('nan'))

    alpha = 1.0 - confidence
    lower = np.percentile(estimates, alpha / 2 * 100)
    upper = np.percentile(estimates, (1 - alpha / 2) * 100)

    return (float(lower), float(upper))

def bootstrap_ci_slope(steps: List[int], accs: List[float], cliff_step: int, n_bootstraps: int = 1000, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Computes bootstrap confidence intervals for the slopes before and after the cliff step.
    """
    try:
        idx = steps.index(cliff_step)
    except ValueError:
        return {'slope_before_ci': (float('nan'), float('nan')), 'slope_after_ci': (float('nan'), float('nan'))}

    x = np.array(steps, dtype=float)
    y = np.array(accs, dtype=float)

    alpha = 1.0 - confidence

    def boot_slope(x_part, y_part):
        n = len(x_part)
        if n < 2:
            return (float('nan'), float('nan'))

        slopes = []
        for _ in range(n_bootstraps):
            indices = np.random.choice(n, size=n, replace=True)
            # Need at least two distinct x values to fit a line
            if len(np.unique(x_part[indices])) > 1:
                s, _, _, _, _ = stats.linregress(x_part[indices], y_part[indices])
                slopes.append(s)

        slopes = np.array(slopes)
        slopes = slopes[~np.isnan(slopes)]

        if len(slopes) == 0:
            return (float('nan'), float('nan'))

        lower = np.percentile(slopes, alpha / 2 * 100)
        upper = np.percentile(slopes, (1 - alpha / 2) * 100)
        return (float(lower), float(upper))

    ci_before = boot_slope(x[:idx+1], y[:idx+1])
    ci_after = boot_slope(x[idx:], y[idx:])

    return {
        'slope_before_ci': ci_before,
        'slope_after_ci': ci_after
    }



def estimate_slopes(steps: List[int], accs: List[float], cliff_step: int) -> Dict[str, float]:
    """
    Estimates the slopes of the accuracy curve before and after the cliff step using linear regression.
    """
    try:
        idx = steps.index(cliff_step)
    except ValueError:
        return {'slope_before': float('nan'), 'slope_after': float('nan')}

    x = np.array(steps, dtype=float)
    y = np.array(accs, dtype=float)

    # Before the cliff
    if idx > 1:
        slope_before, _, _, _, _ = stats.linregress(x[:idx+1], y[:idx+1])
    else:
        slope_before = float('nan')

    # After the cliff
    if idx < len(steps) - 2:
        slope_after, _, _, _, _ = stats.linregress(x[idx:], y[idx:])
    else:
        slope_after = float('nan')

    return {
        'slope_before': slope_before,
        'slope_after': slope_after
    }

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes Cohen's d effect size for two independent samples.
    """
    nx, ny = len(x), len(y)

    if nx < 2 or ny < 2:
        return float('nan')

    dof = nx + ny - 2
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / dof)

    if pooled_std == 0:
        return 0.0

    return (np.mean(x) - np.mean(y)) / pooled_std

def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes the Wilson score interval for a binomial proportion.
    """
    if trials == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials

    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    spread = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator

    return (max(0.0, center - spread), min(1.0, center + spread))

def holm_correction(pvalues: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni correction to a list of p-values.
    """
    n = len(pvalues)
    indices = list(range(n))
    sorted_indices = sorted(indices, key=lambda i: pvalues[i])
    corrected = [0.0] * n
    cumulative_max = 0.0
    for i, idx in enumerate(sorted_indices):
        adj = pvalues[idx] * (n - i)
        cumulative_max = max(cumulative_max, adj)
        corrected[idx] = min(1.0, cumulative_max)
    return corrected

from .cliff_stability import cliffs_delta

def compare_cliff_distributions(steps_group1: List[int], steps_group2: List[int]) -> Dict[str, Any]:
    """
    Returns statistics and box/violin data.
    """
    x = np.array(steps_group1, dtype=float)
    y = np.array(steps_group2, dtype=float)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    nx, ny = len(x), len(y)

    res = {
        'n1': nx,
        'n2': ny,
        'mean1': float(np.mean(x)) if nx > 0 else None,
        'mean2': float(np.mean(y)) if ny > 0 else None,
        'median1': float(np.median(x)) if nx > 0 else None,
        'median2': float(np.median(y)) if ny > 0 else None,
        'cohens_d': float('nan'),
        'cliffs_delta': float('nan'),
        'mwu_statistic': float('nan'),
        'mwu_pvalue': float('nan'),
        'box_data_1': x.tolist(),
        'box_data_2': y.tolist(),
    }

    if nx > 1 and ny > 1:
        res['cohens_d'] = cohens_d(x, y)
        res['cliffs_delta'] = cliffs_delta(x, y)
        mwu = stats.mannwhitneyu(x, y, alternative='two-sided')
        res['mwu_statistic'] = float(mwu.statistic)
        res['mwu_pvalue'] = float(mwu.pvalue)

    return res

def compute_grokking_rates(grokked_group1: List[bool], grokked_group2: List[bool]) -> Dict[str, Any]:
    """
    Computes grokking success rates and binomial confidence intervals.
    """
    n1, n2 = len(grokked_group1), len(grokked_group2)
    s1, s2 = sum(grokked_group1), sum(grokked_group2)

    rate1 = s1 / n1 if n1 > 0 else 0.0
    rate2 = s2 / n2 if n2 > 0 else 0.0

    ci1 = wilson_score_interval(s1, n1)
    ci2 = wilson_score_interval(s2, n2)

    return {
        'rate1': rate1,
        'rate2': rate2,
        'ci1_lower': ci1[0],
        'ci1_upper': ci1[1],
        'ci2_lower': ci2[0],
        'ci2_upper': ci2[1],
        'n1': n1,
        'n2': n2,
    }

def generate_report(comparisons: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Generates a markdown report and a corresponding JSON dictionary.

    Args:
        comparisons: A dictionary mapping comparison names (e.g. 'pure_vs_noise0.15')
                     to a dictionary containing 'cliff_stats' and 'rate_stats'.

    Returns:
        (markdown_string, json_dict)
    """

    lines = ["# Rigorous Stats Report\n\n"]

    for comp_name, data in comparisons.items():
        lines.append(f"## Comparison: {comp_name}\n")

        cstats = data.get('cliff_stats', {})
        rstats = data.get('rate_stats', {})

        lines.append("### Grokking Rates\n")
        lines.append(f"- Group 1: {rstats.get('rate1', 0.0):.2%} (95% CI: [{rstats.get('ci1_lower', 0.0):.2%}, {rstats.get('ci1_upper', 0.0):.2%}], n={rstats.get('n1', 0)})\n")
        lines.append(f"- Group 2: {rstats.get('rate2', 0.0):.2%} (95% CI: [{rstats.get('ci2_lower', 0.0):.2%}, {rstats.get('ci2_upper', 0.0):.2%}], n={rstats.get('n2', 0)})\n\n")

        lines.append("### Cliff Step Distributions (Grokked seeds only)\n")
        if cstats.get('n1', 0) > 0 and cstats.get('n2', 0) > 0:
            lines.append(f"- Group 1 (n={cstats.get('n1')}): Mean = {cstats.get('mean1', 0.0):.1f}, Median = {cstats.get('median1', 0.0):.1f}\n")
            lines.append(f"- Group 2 (n={cstats.get('n2')}): Mean = {cstats.get('mean2', 0.0):.1f}, Median = {cstats.get('median2', 0.0):.1f}\n")

            d_str = f"{cstats.get('cohens_d', float('nan')):.3f}" if not math.isnan(cstats.get('cohens_d', float('nan'))) else "NaN"
            p_str = f"{cstats.get('mwu_pvalue', float('nan')):.3g}" if not math.isnan(cstats.get('mwu_pvalue', float('nan'))) else "NaN"

            lines.append(f"- Effect Size (Cohen's d): {d_str}\n")
            lines.append(f"- Mann-Whitney U test p-value: {p_str}\n")
            lines.append(f"- Data for box/violin plots included in JSON.\n\n")
        else:
            lines.append("- Insufficient data to compare cliff steps.\n\n")

    return "".join(lines), comparisons
