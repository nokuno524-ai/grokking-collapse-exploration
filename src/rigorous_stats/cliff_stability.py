import numpy as np
import json
import csv
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from .cliff_detection import find_cliff_location

def aggregate_cliff_locations(runs_data: List[Dict]) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """
    Computes per-seed cliff locations and aggregates them to return a median and bootstrap CI.

    Args:
        runs_data: A list of dicts, each representing a run. Each run should contain
                   'steps' (list or array) and 'accuracies' (list or array).

    Returns:
        median_cliff: The median cliff location across seeds, or None if no valid cliffs.
        ci: A tuple (lower_bound, upper_bound) representing the 95% bootstrap CI, or None.
    """
    locations = []

    for run in runs_data:
        steps = np.array(run.get("steps", []))
        accuracies = np.array(run.get("accuracies", []))

        if len(steps) == 0 or len(accuracies) == 0:
            continue

        cliff, _ = find_cliff_location(steps, accuracies)
        if cliff is not None:
            locations.append(cliff)

    if not locations:
        return None, None

    locations = np.array(locations)
    median_cliff = float(np.median(locations))

    # Bootstrap CI for the median
    n_boot = 1000
    boot_medians = []
    rng = np.random.RandomState(42)

    for _ in range(n_boot):
        sample = rng.choice(locations, size=len(locations), replace=True)
        boot_medians.append(np.median(sample))

    boot_medians.sort()
    lower = float(boot_medians[int(0.025 * n_boot)])
    upper = float(boot_medians[int(0.975 * n_boot)])

    return median_cliff, (lower, upper)

def export_cliff_summary(results_by_condition: Dict[str, Tuple[Optional[float], Optional[Tuple[float, float]]]], out_csv: str, out_json: str):
    """
    Exports the aggregated cliff locations and CIs to CSV and JSON formats.

    Args:
        results_by_condition: A dictionary mapping condition names to (median, ci) tuples.
        out_csv: Path to the output CSV file.
        out_json: Path to the output JSON file.
    """
    # JSON export
    json_data = {}
    for condition, (median, ci) in results_by_condition.items():
        json_data[condition] = {
            "median_cliff": median,
            "ci_lower": ci[0] if ci else None,
            "ci_upper": ci[1] if ci else None
        }

    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=2)

    # CSV export
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "median_cliff", "ci_lower", "ci_upper"])
        for condition, (median, ci) in results_by_condition.items():
            lower = ci[0] if ci else ""
            upper = ci[1] if ci else ""
            med_str = median if median is not None else ""
            writer.writerow([condition, med_str, lower, upper])
