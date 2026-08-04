import json
import numpy as np
from pathlib import Path
from scipy import stats
import pandas as pd

def cohen_d(x, y):
    """Calculate Cohen's d effect size."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return 0.0
    # pooled standard deviation
    pool_sd = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    if pool_sd == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pool_sd

def bootstrap_ci(data, stat_func=np.mean, n_bootstraps=1000, ci=95):
    """Calculate bootstrap confidence interval."""
    bootstrapped_stats = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_stats.append(stat_func(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrapped_stats, alpha)
    upper = np.percentile(bootstrapped_stats, 100 - alpha)
    return np.mean(bootstrapped_stats), lower, upper

def compare_conditions(cond1, cond2, results_dir=Path('results/multi_seed')):
    """Compare two conditions using Mann-Whitney U test."""
    def get_grokking_steps(cond):
        steps = []
        for seed_dir in results_dir.glob('*'):
            if not seed_dir.is_dir(): continue
            path = seed_dir / cond / 'results.json'
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    grok = data.get('grokking_step')
                    if grok is not None:
                        steps.append(grok)
                    else:
                        steps.append(50000) # Max steps if didn't grok
        return steps

    steps1 = get_grokking_steps(cond1)
    steps2 = get_grokking_steps(cond2)

    if not steps1 or not steps2:
        return None

    u_stat, p_val = stats.mannwhitneyu(steps1, steps2, alternative='two-sided')
    effect = cohen_d(steps1, steps2)

    mean1, lower1, upper1 = bootstrap_ci(steps1)
    mean2, lower2, upper2 = bootstrap_ci(steps2)

    return {
        'cond1': cond1,
        'cond2': cond2,
        'u_stat': u_stat,
        'p_value': p_val,
        'cohen_d': effect,
        'cond1_mean': mean1,
        'cond1_ci': (lower1, upper1),
        'cond2_mean': mean2,
        'cond2_ci': (lower2, upper2),
    }

def generate_latex_table(comparisons, output_path='results/statistical_analysis.tex'):
    """Generate LaTeX table for statistical results."""
    latex = """\\begin{table}[h]
\\centering
\\begin{tabular}{llcccc}
\\toprule
\\textbf{Condition 1} & \\textbf{Condition 2} & \\textbf{Cohen's d} & \\textbf{p-value} & \\textbf{U-Stat} & \\textbf{Significance} \\\\
\\midrule
"""
    for comp in comparisons:
        if comp is None: continue
        sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else "ns"
        c1 = comp['cond1'].replace('_', '\\_')
        c2 = comp['cond2'].replace('_', '\\_')
        latex += f"{c1} & {c2} & {comp['cohen_d']:.2f} & {comp['p_value']:.4f} & {comp['u_stat']:.1f} & {sig} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\caption{Statistical comparison of grokking steps between conditions (Mann-Whitney U test).}
\\label{tab:grokking_stats}
\\end{table}
"""
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved to {output_path}")

if __name__ == "__main__":
    np.random.seed(42)
    conditions = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]
    comparisons = []

    # Compare each collapse condition to pure
    for cond in conditions[1:]:
        comp = compare_conditions("pure", cond)
        if comp:
            comparisons.append(comp)

    # Also compare low to medium, medium to high
    comparisons.append(compare_conditions("low_collapse", "medium_collapse"))
    comparisons.append(compare_conditions("medium_collapse", "high_collapse"))

    generate_latex_table(comparisons)
