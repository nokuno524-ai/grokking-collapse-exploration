"""
Statistical analysis tools for evaluating grokking and collapse metrics.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Any

def compute_statistics(data: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Compute mean, std, median, and confidence interval for a list of numbers."""
    if not data:
        return {"mean": float('nan'), "std": float('nan'), "median": float('nan'), "ci_lower": float('nan'), "ci_upper": float('nan')}

    a = 1.0 * np.array(data)
    n = len(a)
    mean = np.mean(a)
    std = np.std(a, ddof=1) if n > 1 else 0.0
    median = np.median(a)

    if n > 1:
        se = std / np.sqrt(n)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        ci_lower = mean - h
        ci_upper = mean + h
    else:
        ci_lower = mean
        ci_upper = mean

    return {
        "mean": float(mean),
        "std": float(std),
        "median": float(median),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper)
    }

def compare_conditions(condition_a: List[float], condition_b: List[float]) -> Dict[str, float]:
    """
    Compare two conditions using an independent t-test and calculate Cohen's d.
    Returns t-statistic, p-value, and Cohen's d effect size.
    """
    if len(condition_a) < 2 or len(condition_b) < 2:
        return {"t_stat": float('nan'), "p_value": float('nan'), "cohens_d": float('nan')}

    a = np.array(condition_a)
    b = np.array(condition_b)

    # Welch's t-test (assumes unequal variances)
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    # Cohen's d
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)

    if pooled_var == 0:
        d = float('inf') if np.mean(a) != np.mean(b) else 0.0
    else:
        d = (np.mean(a) - np.mean(b)) / np.sqrt(pooled_var)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(d)
    }

def anova_across_conditions(*conditions: List[float]) -> Dict[str, float]:
    """
    Perform a one-way ANOVA across multiple conditions.
    Requires at least 2 conditions, and each must have at least 2 data points.
    """
    valid_conditions = [c for c in conditions if len(c) >= 2]
    if len(valid_conditions) < 2:
        return {"f_stat": float('nan'), "p_value": float('nan')}

    f_stat, p_value = stats.f_oneway(*valid_conditions)
    return {
        "f_stat": float(f_stat),
        "p_value": float(p_value)
    }

def bootstrap_ci(data: List[float], num_resamples: int = 10000, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval using bootstrap resampling.
    Useful for non-normal distributions or small sample sizes.
    """
    if len(data) < 2:
        val = data[0] if data else float('nan')
        return (val, val)

    a = np.array(data)
    resamples = np.random.choice(a, size=(num_resamples, len(a)), replace=True)
    means = np.mean(resamples, axis=1)

    lower_percentile = (1.0 - confidence) / 2.0 * 100
    upper_percentile = (1.0 + confidence) / 2.0 * 100

    lower_bound = np.percentile(means, lower_percentile)
    upper_bound = np.percentile(means, upper_percentile)

    return float(lower_bound), float(upper_bound)

def generate_stat_report(results_dict: Dict[str, Dict[str, List[float]]]) -> str:
    """
    Generate a LaTeX table reporting the statistical results.
    results_dict format: { 'TaskName': { 'ConditionName': [list_of_metric_values] } }
    """
    latex_str = "\\begin{table}[h]\n\\centering\n"
    latex_str += "\\begin{tabular}{llccc}\n\\toprule\n"
    latex_str += "Task & Condition & Mean & Std & 95\\% CI \\\\\n\\midrule\n"

    for task_name, conditions in results_dict.items():
        for i, (cond_name, data) in enumerate(conditions.items()):
            stats_res = compute_statistics(data)
            mean = stats_res['mean']
            std = stats_res['std']
            ci_l, ci_u = stats_res['ci_lower'], stats_res['ci_upper']

            # Use multi-row conceptually by only printing task name on first row
            task_disp = task_name.replace('_', '\\_') if i == 0 else ""
            cond_disp = str(cond_name).replace('_', '\\_')

            latex_str += f"{task_disp} & {cond_disp} & {mean:.3f} & {std:.3f} & [{ci_l:.3f}, {ci_u:.3f}] \\\\\n"
        latex_str += "\\midrule\n"

    latex_str += "\\bottomrule\n\\end{tabular}\n"
    latex_str += "\\caption{Statistical summary of metrics across tasks and conditions.}\n"
    latex_str += "\\label{tab:stats}\n\\end{table}"

    return latex_str
