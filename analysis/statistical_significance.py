import numpy as np
import pandas as pd
from scipy import stats
import json
from pathlib import Path
from typing import Dict, List

def bootstrap_ci(data: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95):
    """Compute BCa bootstrap confidence interval for the mean. Needs variance in data."""
    if len(data) < 2:
        return (data[0], data[0]) if len(data) == 1 else (0.0, 0.0)
    if np.var(data) == 0:
        return (data[0], data[0])

    res = stats.bootstrap((data,), np.mean, confidence_level=confidence,
                          n_resamples=n_resamples, method='BCa')
    return (res.confidence_interval.low, res.confidence_interval.high)

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_sd

def perform_tests(results_dir="results/phase_transitions") -> Dict:
    """Load results, perform Mann-Whitney U tests and compute effect sizes."""
    p = Path(results_dir)
    data = []
    if not p.exists():
        return {}

    for cond_dir in p.iterdir():
        if not cond_dir.is_dir(): continue
        for seed_dir in cond_dir.iterdir():
            if not seed_dir.is_dir(): continue
            res_file = seed_dir / "results.json"
            if res_file.exists():
                try:
                    with open(res_file, 'r') as f:
                        res = json.load(f)
                    cfg = res['config']
                    data.append({
                        'noise': cfg.get('noise_fraction', 0),
                        'collapse': cfg.get('collapse_level', 0),
                        'wd': cfg.get('weight_decay', 1.0),
                        'test_acc': res.get('final_test_acc', 0),
                        'fourier': res.get('final_fourier_concentration', 0),
                        'grokked': 1 if res.get('grokked', False) else 0
                    })
                except Exception:
                    pass

    df = pd.DataFrame(data)
    if df.empty:
        return {}

    results = {}

    # Compare Noise=0.15 vs Collapse=0.15 at wd=1.0
    wd1_df = df[df['wd'] == 1.0]
    pure_noise = wd1_df[(wd1_df['noise'] == 0.15) & (wd1_df['collapse'] == 0.0)]['test_acc'].values
    pure_collapse = wd1_df[(wd1_df['noise'] == 0.0) & (wd1_df['collapse'] == 0.15)]['test_acc'].values

    if len(pure_noise) > 0 and len(pure_collapse) > 0:
        # Check if identical (can't run MWU meaningfully if they are exactly identical distributions that are constant)
        if np.array_equal(pure_noise, pure_collapse) and np.var(pure_noise) == 0:
            stat, p_val = 0.0, 1.0
        else:
            stat, p_val = stats.mannwhitneyu(pure_noise, pure_collapse, alternative='two-sided')

        d = cohens_d(pure_noise, pure_collapse)
        results['noise_vs_collapse_acc'] = {
            'mann_whitney_u': stat,
            'p_value': p_val,
            'cohens_d': d
        }

    return results

def write_publication_table(results: Dict, output_path="analysis/statistical_summary.tex"):
    """Write results to a publication-ready LaTeX table."""
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    tex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Comparison & MWU Stat & $p$-value & Cohen's $d$ \\\\",
        "\\midrule"
    ]

    for key, val in results.items():
        name = key.replace('_', ' ').title()
        tex.append(f"{name} & {val['mann_whitney_u']:.2f} & {val['p_value']:.4f} & {val['cohens_d']:.2f} \\\\")

    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Statistical significance tests for Grokking phase transitions.}",
        "\\end{table}"
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(tex))

if __name__ == "__main__":
    res = perform_tests()
    if res:
        write_publication_table(res)
        print("Wrote statistical summary to analysis/statistical_summary.tex")
    else:
        print("No results found to process.")
