import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import List, Dict, Any, Tuple, Optional

def logistic(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """Logistic function for curve fitting."""
    return L / (1 + np.exp(np.clip(-k * (x - x0), -500, 500))) + b

def fit_severity_relationship(severities: np.ndarray, cliffs: np.ndarray) -> Dict[str, Any]:
    """
    Fits both a linear and logistic curve to severity vs cliff step.
    Returns parameters and R-squared for both.
    """
    results = {}

    # Sort data
    idx = np.argsort(severities)
    x = severities[idx]
    y = cliffs[idx]

    if len(x) < 3:
        return results

    # Linear fit
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        results['linear'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r2': float(r_value**2),
            'p_value': float(p_value),
            'std_err': float(std_err)
        }
    except Exception as e:
        pass

    # Logistic fit
    try:
        # Initial guess: L=max-min, k=mean slope, x0=median x, b=min
        p0 = [np.max(y)-np.min(y), 1.0, np.median(x), np.min(y)]

        # Add a tiny bit of jitter to avoid singular matrices if data points have same x
        x_jitter = x + np.random.normal(0, 1e-6, size=x.shape)

        popt, pcov = curve_fit(logistic, x_jitter, y, p0=p0, maxfev=10000)

        # Compute R2
        residuals = y - logistic(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        results['logistic'] = {
            'L': float(popt[0]),
            'k': float(popt[1]),
            'x0': float(popt[2]),
            'b': float(popt[3]),
            'r2': float(r2),
            # diag of covariance matrix gives param variance
            'param_std': [float(v) for v in np.sqrt(np.diag(pcov))]
        }
    except Exception as e:
        pass

    return results

def holm_correction(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni correction for multiple comparisons.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values while keeping track of original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    adj_p = np.zeros(m)
    for i, p in enumerate(sorted_p):
        correction_factor = m - i
        adj_p[i] = min(1.0, p * correction_factor)

        # Enforce monotonicity
        if i > 0 and adj_p[i] < adj_p[i-1]:
            adj_p[i] = adj_p[i-1]

    # Revert to original order
    final_adj_p = np.zeros(m)
    for i, orig_idx in enumerate(sorted_idx):
        final_adj_p[orig_idx] = adj_p[i]

    return final_adj_p.tolist()

def compare_endpoints(grouped_runs: Dict[float, List[Dict[str, Any]]], baseline_severity: float = 0.0) -> Dict[str, Any]:
    """
    Compares final accuracy distributions across severities using Mann-Whitney U,
    with Holm correction.
    """
    if baseline_severity not in grouped_runs:
        return {}

    baseline_accs = []
    for run in grouped_runs[baseline_severity]:
        if run['history']:
            baseline_accs.append(run['history'][-1]['test_acc'])

    if len(baseline_accs) == 0:
        return {}

    comparisons = {}
    p_values = []
    severities = []

    for severity, runs in grouped_runs.items():
        if severity == baseline_severity:
            continue

        accs = []
        for run in runs:
            if run['history']:
                accs.append(run['history'][-1]['test_acc'])

        if len(accs) == 0:
            continue

        try:
            stat, p = stats.mannwhitneyu(baseline_accs, accs, alternative='two-sided')
            if np.isnan(p):
                p = 1.0 # edge case where arrays are identical
        except Exception:
            p = 1.0

        comparisons[severity] = {
            'mean_acc': float(np.mean(accs)),
            'median_acc': float(np.median(accs)),
            'raw_p_value': float(p)
        }

        p_values.append(p)
        severities.append(severity)

    # Apply correction
    adj_p_values = holm_correction(p_values)

    for sev, adj_p in zip(severities, adj_p_values):
        comparisons[sev]['adj_p_value'] = adj_p

    return {
        'baseline_severity': baseline_severity,
        'baseline_mean': float(np.mean(baseline_accs)),
        'baseline_median': float(np.median(baseline_accs)),
        'comparisons': comparisons
    }
