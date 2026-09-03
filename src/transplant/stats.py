import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import scipy.stats as stats

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size for two independent samples."""
    nx = len(x)
    ny = len(y)

    if nx <= 1 or ny <= 1:
        return 0.0

    dof = nx + ny - 2

    # Pooled standard deviation
    pool_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    pool_sd = np.sqrt(pool_var)

    if pool_sd == 0:
        return 0.0

    return (np.mean(x) - np.mean(y)) / pool_sd

def bootstrap_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    n = len(data)

    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return data[0], data[0]

    samples = rng.choice(data, size=(n_boot, n), replace=True)
    means = np.mean(samples, axis=1)

    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100

    return np.percentile(means, lower_pct), np.percentile(means, upper_pct)

def sign_consistency(data: np.ndarray) -> float:
    """Returns the fraction of items with the same sign as the mean."""
    mean = np.mean(data)
    if mean > 0:
        return np.mean(data > 0)
    elif mean < 0:
        return np.mean(data < 0)
    return 0.5

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-seed transplant results and compute statistics."""
    records = []

    grouped = df.groupby(['donor_condition', 'recipient_condition', 'layer_idx', 'head_idx', 'component_type'], dropna=False)

    for name, group in grouped:
        donor_cond, recip_cond, l_idx, h_idx, comp_type = name

        acc_deltas = group['acc_delta'].values
        mean_delta = np.mean(acc_deltas)
        ci_lower, ci_upper = bootstrap_ci(acc_deltas)

        # We compare transplant_acc with baseline_acc
        d = cohens_d(group['transplant_acc'].values, group['baseline_acc'].values)

        # P-value from paired t-test
        try:
            _, p_val = stats.ttest_rel(group['transplant_acc'].values, group['baseline_acc'].values)
        except:
            p_val = 1.0

        if np.isnan(p_val):
            p_val = 1.0

        consistency = sign_consistency(acc_deltas)

        # Aggregate constant attention check (if it was True for ANY seed, we might flag it, or majority)
        const_attn_frac = group['is_constant_attention'].mean() if 'is_constant_attention' in group.columns else 0.0

        records.append({
            'donor_condition': donor_cond,
            'recipient_condition': recip_cond,
            'layer_idx': l_idx,
            'head_idx': h_idx,
            'component_type': comp_type,
            'mean_acc_delta': mean_delta,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'cohens_d': d,
            'p_value': p_val,
            'sign_consistency': consistency,
            'is_constant_attention_frac': const_attn_frac,
            'n_seeds': len(group)
        })

    return pd.DataFrame(records)
