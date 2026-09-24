import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

def compute_bootstrap_ci(data: List[float], num_samples: int = 1000) -> Tuple[float, float, float]:
    """
    Compute the median and its 95% bootstrap confidence interval.

    Args:
        data: List of numerical values.
        num_samples: Number of bootstrap samples to generate.

    Returns:
        tuple: (median, lower_ci, upper_ci)
    """
    data = np.array(data)
    if len(data) == 0:
        return float('nan'), float('nan'), float('nan')

    median = np.median(data)

    # Bootstrap
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(data), size=(num_samples, len(data)))
    samples = data[indices]
    medians = np.median(samples, axis=1)

    lower = np.percentile(medians, 2.5)
    upper = np.percentile(medians, 97.5)

    return float(median), float(lower), float(upper)

def compute_survival_analysis(runs: List[Dict[str, Any]], max_steps: int = 50000) -> Dict[str, Any]:
    """
    Compute right-censored median estimates for survival curves of grokking.

    Args:
        runs: List of run dictionaries, each expected to have a 'grokking_step'.
        max_steps: The maximum steps run (censoring time).

    Returns:
        Dict: Median survival step and confidence intervals if available.
    """
    # Extract grokking steps, applying max_steps for censored data
    times = []
    events = []

    for run in runs:
        g_step = run.get('grokking_step')
        if g_step is not None:
            times.append(g_step)
            events.append(1) # event occurred
        else:
            times.append(max_steps)
            events.append(0) # right-censored

    times = np.array(times)
    events = np.array(events)

    if len(times) == 0:
        return {'median': float('nan'), 'lower': float('nan'), 'upper': float('nan')}

    # Standard median among those that did grok
    grok_times = times[events == 1]
    if len(grok_times) > 0:
        med, lower, upper = compute_bootstrap_ci(grok_times.tolist())
    else:
        med, lower, upper = float('nan'), float('nan'), float('nan')

    return {
        'times': times.tolist(),
        'events': events.tolist(),
        'median': med,
        'lower_ci': lower,
        'upper_ci': upper,
        'grok_fraction': float(np.mean(events))
    }

def permutation_test(group_a: List[float], group_b: List[float], n_permutations: int = 10000) -> float:
    """
    Perform a permutation test to evaluate the difference in medians between two groups.

    Args:
        group_a: First group of values.
        group_b: Second group of values.
        n_permutations: Number of permutations.

    Returns:
        float: p-value of the test.
    """
    group_a = np.array(group_a)
    group_b = np.array(group_b)

    if len(group_a) == 0 or len(group_b) == 0:
        return float('nan')

    obs_diff = abs(np.median(group_a) - np.median(group_b))

    combined = np.concatenate([group_a, group_b])
    rng = np.random.default_rng(42)

    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(combined)
        new_a = shuffled[:len(group_a)]
        new_b = shuffled[len(group_a):]
        diff = abs(np.median(new_a) - np.median(new_b))
        if diff >= obs_diff:
            count += 1

    return count / n_permutations

def generate_stats_report(results_dir: str, output_dir: str) -> None:
    """
    Iterate over conditions in multi-seed results, compute stats, and generate report.

    Args:
        results_dir: Path to multi-seed results directory.
        output_dir: Output directory for reports.
    """
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    conditions = {}

    # Structure is results/multi_seed/<seed>/<condition>/results.json
    for seed_dir in results_path.iterdir():
        if not seed_dir.is_dir():
            continue
        for cond_dir in seed_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            cond_name = cond_dir.name
            res_file = cond_dir / "results.json"
            if res_file.exists():
                with open(res_file, "r") as f:
                    data = json.load(f)
                if cond_name not in conditions:
                    conditions[cond_name] = []
                conditions[cond_name].append(data)


    report = {}

    # Calculate basic survival stats
    for cond_name, runs in conditions.items():
        surv_stats = compute_survival_analysis(runs)
        report[cond_name] = surv_stats

    # Calculate permutation p-values vs pure
    if 'pure' in report:
        pure_times = []
        for run in conditions['pure']:
            g_step = run.get('grokking_step')
            if g_step is not None: pure_times.append(g_step)

        for cond_name in report:
            if cond_name == 'pure':
                report[cond_name]['p_value_vs_pure'] = 1.0
                continue

            cond_times = []
            for run in conditions[cond_name]:
                g_step = run.get('grokking_step')
                if g_step is not None: cond_times.append(g_step)

            pval = permutation_test(pure_times, cond_times)
            report[cond_name]['p_value_vs_pure'] = float(pval) if not np.isnan(pval) else None

    with open(out_path / "stats_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out_path / "stats_report.md", "w") as f:
        f.write("# Rigorous Statistical Analysis Report\n\n")
        f.write("| Condition | Grok Fraction | Median Step | 95% CI Lower | 95% CI Upper | p-value vs Pure |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cond_name, stats in report.items():
            pval = stats.get('p_value_vs_pure')
            pval_str = f"{pval:.4f}" if pval is not None else "N/A"
            f.write(f"| {cond_name} | {stats['grok_fraction']:.2f} | {stats['median']:.1f} | {stats['lower_ci']:.1f} | {stats['upper_ci']:.1f} | {pval_str} |\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        generate_stats_report(sys.argv[1], sys.argv[2])
