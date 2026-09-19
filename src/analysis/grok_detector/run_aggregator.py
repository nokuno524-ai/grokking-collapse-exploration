import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import csv

from .detectors import max_2nd_derivative_detector
from .stats import compute_median_iqr, holm_mann_whitney, permutation_spearman


def parse_multi_seed_results(results_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parses the results/multi_seed/<seed>/<condition>/results.json structure.

    Returns:
        A dictionary mapping condition names to a list of run dictionaries.
    """
    runs_by_condition = {}
    base_path = Path(results_dir)

    if not base_path.exists():
        return runs_by_condition

    for seed_dir in base_path.iterdir():
        if not seed_dir.is_dir():
            continue

        seed = seed_dir.name

        for cond_dir in seed_dir.iterdir():
            if not cond_dir.is_dir():
                continue

            cond_name = cond_dir.name
            results_file = cond_dir / "results.json"

            if results_file.exists():
                with open(results_file, 'r') as f:
                    try:
                        data = json.load(f)
                        data['seed'] = seed
                        data['condition'] = cond_name

                        if cond_name not in runs_by_condition:
                            runs_by_condition[cond_name] = []
                        runs_by_condition[cond_name].append(data)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {results_file}")

    return runs_by_condition


def aggregate_and_plot(
    results_dir: str,
    output_csv: str = "results_aggregated.csv",
    window_size: int = 5,
    control_condition: str = "pure"
) -> Dict[str, Any]:
    """
    Aggregates multi-seed results, applies change-point detection, and computes statistics.

    Args:
        results_dir: Path to the multi_seed results directory.
        output_csv: Path to save the aggregated CSV.
        window_size: Smoothing window size for cliff detection.
        control_condition: The condition to compare others against for statistical tests.

    Returns:
        A dictionary with the computed statistics.
    """
    runs_by_condition = parse_multi_seed_results(results_dir)

    if not runs_by_condition:
        print(f"No results found in {results_dir}")
        return {}

    # Apply change-point detection to each run
    all_runs_flat = []
    condition_cliff_steps = {}  # For stats

    for cond_name, runs in runs_by_condition.items():
        condition_cliff_steps[cond_name] = []

        for run in runs:
            history = run.get('history', [])
            steps = []
            test_accs = []

            for entry in history:
                if 'step' in entry and 'test_acc' in entry and not np.isnan(entry['test_acc']):
                    steps.append(entry['step'])
                    test_accs.append(entry['test_acc'])

            cliff_result = max_2nd_derivative_detector(steps, test_accs, window_size=window_size)

            run['computed_grokked'] = cliff_result['grokked']
            run['computed_cliff_step'] = cliff_result['cliff_step']
            run['computed_peak_width'] = cliff_result['peak_width']

            if cliff_result['grokked'] and cliff_result['cliff_step'] is not None:
                condition_cliff_steps[cond_name].append(cliff_result['cliff_step'])

            all_runs_flat.append(run)

    # Compute per-condition statistics
    stats_by_condition = {}
    for cond_name, cliff_steps in condition_cliff_steps.items():
        total_runs = len(runs_by_condition[cond_name])
        grokked_runs = len(cliff_steps)
        frac_grokked = grokked_runs / total_runs if total_runs > 0 else 0

        median, q25, q75 = compute_median_iqr(cliff_steps)

        stats_by_condition[cond_name] = {
            'total_runs': total_runs,
            'grokked_runs': grokked_runs,
            'fraction_grokked': frac_grokked,
            'median_cliff_step': median,
            'iqr_25_cliff_step': q25,
            'iqr_75_cliff_step': q75,
        }

    # Pairwise statistical tests
    mann_whitney_results = {}
    if control_condition in condition_cliff_steps:
        mann_whitney_results = holm_mann_whitney(condition_cliff_steps, control_condition)

    # Correlation between weight norm and cliff presence
    all_weight_norms = []
    all_grokked_flags = []

    for run in all_runs_flat:
        final_wn = run.get('final_weight_norm')
        # if final_weight_norm is not there, try to get from history
        if final_wn is None or np.isnan(final_wn):
            history = run.get('history', [])
            if history and 'weight_norm' in history[-1]:
                final_wn = history[-1]['weight_norm']

        if final_wn is not None and not np.isnan(final_wn):
            all_weight_norms.append(final_wn)
            all_grokked_flags.append(1.0 if run['computed_grokked'] else 0.0)

    wn_corr, wn_pval = np.nan, np.nan
    if len(all_weight_norms) >= 2:
        wn_corr, wn_pval = permutation_spearman(all_weight_norms, all_grokked_flags)

    # Save to CSV
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            fieldnames = ['condition', 'seed', 'grokked', 'cliff_step', 'peak_width', 'final_test_acc', 'final_weight_norm']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for run in all_runs_flat:
                writer.writerow({
                    'condition': run.get('condition', ''),
                    'seed': run.get('seed', ''),
                    'grokked': run.get('computed_grokked', False),
                    'cliff_step': run.get('computed_cliff_step', ''),
                    'peak_width': run.get('computed_peak_width', ''),
                    'final_test_acc': run.get('final_test_acc', ''),
                    'final_weight_norm': run.get('final_weight_norm', '')
                })

    return {
        'condition_stats': stats_by_condition,
        'mann_whitney': mann_whitney_results,
        'weight_norm_correlation': {
            'spearman_rho': wn_corr,
            'permutation_p_value': wn_pval
        }
    }
