"""
Censoring-aware aggregation for ensemble sweeps.
Treats runs that never grokked as right-censored.
Computes Kaplan-Meier survival curves and summary statistics.
"""

import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add src to python path so we can import utils
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.ensemble.utils import load_run_results


def compute_kaplan_meier(events, max_time):
    """
    Compute Kaplan-Meier survival curve.

    events: list of tuples (time, grokked_bool)
    max_time: max possible step, used to extend curve if needed

    Returns:
        times: list of unique event times
        survival_probs: list of survival probabilities
        variances: list of variance estimates (Greenwood's formula)
    """
    # Sort by time
    events.sort(key=lambda x: x[0])

    n_at_risk = len(events)
    current_time = 0
    current_prob = 1.0
    current_variance = 0.0

    times = [0]
    survival_probs = [1.0]
    variances = [0.0]

    sum_var_terms = 0.0

    i = 0
    while i < len(events):
        t = events[i][0]

        # Count events and censorings at time t
        d_i = 0 # grokked
        c_i = 0 # censored

        while i < len(events) and events[i][0] == t:
            if events[i][1]:
                d_i += 1
            else:
                c_i += 1
            i += 1

        if d_i > 0:
            prob_survival_at_t = 1.0 - (d_i / n_at_risk)
            current_prob *= prob_survival_at_t

            # Greenwood's formula
            if n_at_risk > d_i:
                sum_var_terms += d_i / (n_at_risk * (n_at_risk - d_i))
                current_variance = (current_prob ** 2) * sum_var_terms

        times.append(t)
        survival_probs.append(current_prob)
        variances.append(current_variance)

        n_at_risk -= (d_i + c_i)

    if times[-1] < max_time:
        times.append(max_time)
        survival_probs.append(current_prob)
        variances.append(current_variance)

    return times, survival_probs, variances

def estimate_median_survival(times, survival_probs):
    """Estimate median survival time (when prob falls to 0.5)."""
    for i in range(len(times)):
        if survival_probs[i] <= 0.5:
            return times[i]
    return float('inf')  # Median not reached

def aggregate_results(base_dir: str, max_steps: int = 50000) -> dict:
    """
    Aggregate results from all runs in base_dir.
    Returns a dictionary of conditions and their aggregated statistics.
    """
    # Group by condition
    condition_data = defaultdict(list)

    for result in load_run_results(base_dir):
        config = result.get("config", {})
        condition = config.get("condition_name", "unknown")

        grokked = result.get("grokked", False)
        grok_step = result.get("grokking_step")

        # If grokked, time is grok_step. If not, it's censored at max_steps
        if grokked and grok_step is not None:
            time = grok_step
        else:
            time = max_steps

        condition_data[condition].append({
            "seed": config.get("seed", -1),
            "grokked": grokked,
            "time": time,
            "final_test_acc": result.get("final_test_acc", 0.0)
        })

    # Calculate stats per condition
    aggregated = {}
    for condition, runs in condition_data.items():
        events = [(run["time"], run["grokked"]) for run in runs]
        times, probs, variances = compute_kaplan_meier(events, max_steps)

        median_grok = estimate_median_survival(times, probs)
        grok_fraction = sum(1 for r in runs if r["grokked"]) / len(runs)

        aggregated[condition] = {
            "n_runs": len(runs),
            "n_grokked": sum(1 for r in runs if r["grokked"]),
            "grok_fraction": grok_fraction,
            "median_grok_step": median_grok,
            "km_times": times,
            "km_probs": probs,
            "km_variances": variances,
            "raw_runs": runs
        }

    return aggregated

def export_tidy_csv(aggregated: dict, output_path: str):
    """Export run-level summaries to a tidy CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "grokked", "grok_step", "final_test_acc"])

        for condition, stats in aggregated.items():
            for run in stats["raw_runs"]:
                writer.writerow([
                    condition,
                    run["seed"],
                    run["grokked"],
                    run["time"] if run["grokked"] else "",
                    run["final_test_acc"]
                ])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results directory")
    parser.add_argument("--output-csv", type=str, default="analysis/ensemble_summary.csv", help="Output CSV path")
    parser.add_argument("--max-steps", type=int, default=50000, help="Max steps for censoring")
    args = parser.parse_args()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    agg = aggregate_results(args.results_dir, args.max_steps)
    export_tidy_csv(agg, args.output_csv)

    print(f"Aggregated {sum(s['n_runs'] for s in agg.values())} runs across {len(agg)} conditions.")
    print(f"Saved tidy summary to {args.output_csv}")
