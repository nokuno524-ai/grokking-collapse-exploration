import json
from pathlib import Path
from typing import Dict, List, Any, Optional

def get_severity_order() -> List[str]:
    return ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]

def aggregate_runs(results_dir: Path, max_steps: int = 50000) -> Dict[str, List[Dict[str, Any]]]:
    """
    Aggregates multi-seed results from `results/multi_seed/<seed>/<condition>/results.json`
    Returns a dictionary mapping condition severity to a list of seed results.
    """
    aggregated: Dict[str, List[Dict[str, Any]]] = {cond: [] for cond in get_severity_order()}

    if not results_dir.exists():
        return aggregated

    # Iterate over seeds
    for seed_dir in results_dir.iterdir():
        if not seed_dir.is_dir():
            continue

        seed_name = seed_dir.name

        # Iterate over conditions
        for cond_dir in seed_dir.iterdir():
            if not cond_dir.is_dir():
                continue

            cond_name = cond_dir.name
            results_file = cond_dir / "results.json"

            if not results_file.exists():
                continue

            if cond_name not in aggregated:
                aggregated[cond_name] = []

            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)

                grok_step = data.get("grokking_step")

                # Check right-censoring
                # If grokking_step is None but model trained to max_steps without grokking
                if not data.get("grokked", False):
                    # right censored at max_steps
                    time_to_grok = max_steps
                    event_observed = False
                else:
                    time_to_grok = grok_step
                    event_observed = True

                aggregated[cond_name].append({
                    "seed": seed_name,
                    "grokked": data.get("grokked", False),
                    "grokking_step": grok_step,
                    "time_to_grok": time_to_grok,
                    "event_observed": event_observed,
                    "final_test_acc": data.get("final_test_acc"),
                    "history": data.get("history", [])
                })
            except Exception as e:
                print(f"Error loading {results_file}: {e}")

    return aggregated
