import os
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

from .detectors import max_2nd_derivative_detector
from .run_aggregator import parse_multi_seed_results

def run_sensitivity_analysis(
    results_dir: str,
    output_file: str = "sensitivity_report.json"
) -> Dict[str, Any]:
    """
    Performs a sensitivity analysis by varying the smoothing window size and
    simulating different evaluation cadences on existing results.
    """
    runs_by_condition = parse_multi_seed_results(results_dir)

    if not runs_by_condition:
        return {"error": "No data found"}

    windows_to_test = [1, 3, 5, 7, 10, 15]
    cadence_multipliers = [1, 2, 5, 10]  # e.g., if original is 100, x2 means 200

    sensitivity_results = {}

    for cond_name, runs in runs_by_condition.items():
        cond_stats = {}
        for w in windows_to_test:
            for c_mult in cadence_multipliers:
                key = f"window_{w}_cadence_{c_mult}"
                cliff_steps = []

                for run in runs:
                    history = run.get('history', [])
                    steps = []
                    test_accs = []

                    # Simulate cadence by subsampling
                    for i, entry in enumerate(history):
                        if i % c_mult == 0 and 'step' in entry and 'test_acc' in entry and not np.isnan(entry['test_acc']):
                            steps.append(entry['step'])
                            test_accs.append(entry['test_acc'])

                    cliff_result = max_2nd_derivative_detector(steps, test_accs, window_size=w)
                    if cliff_result['grokked'] and cliff_result['cliff_step'] is not None:
                        cliff_steps.append(cliff_result['cliff_step'])

                # Compute median for this parameter combination
                if cliff_steps:
                    cond_stats[key] = {
                        "median_cliff_step": float(np.median(cliff_steps)),
                        "grokked_fraction": len(cliff_steps) / len(runs)
                    }
                else:
                    cond_stats[key] = {
                        "median_cliff_step": None,
                        "grokked_fraction": 0.0
                    }

        sensitivity_results[cond_name] = cond_stats

    # Analyze robustness (max variation across parameters for median cliff step)
    robustness_summary = {}
    for cond_name, stats in sensitivity_results.items():
        medians = [s["median_cliff_step"] for s in stats.values() if s["median_cliff_step"] is not None]
        if medians:
            robustness_summary[cond_name] = {
                "min_median": min(medians),
                "max_median": max(medians),
                "range": max(medians) - min(medians),
                "std_dev": float(np.std(medians))
            }
        else:
            robustness_summary[cond_name] = "No grokking detected in any configuration"

    final_report = {
        "parameters_tested": {
            "windows": windows_to_test,
            "cadence_multipliers": cadence_multipliers
        },
        "robustness_summary": robustness_summary,
        "detailed_results": sensitivity_results
    }

    with open(output_file, 'w') as f:
        json.dump(final_report, f, indent=2, sort_keys=True)

    return final_report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/multi_seed")
    parser.add_argument("--output", type=str, default="results/multi_seed/sensitivity_report.json")
    args = parser.parse_args()

    print(f"Running sensitivity analysis on {args.results_dir}...")
    report = run_sensitivity_analysis(args.results_dir, args.output)
    print(json.dumps(report["robustness_summary"], indent=2))
    print(f"Full report saved to {args.output}")
