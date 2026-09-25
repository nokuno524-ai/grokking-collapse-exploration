"""
Utilities for loading and processing ensemble run results.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

def load_run_results(results_dir: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load multi-seed results from a hierarchical directory structure.
    Expected structure: results_dir/<seed>/<condition>/results.json

    Args:
        results_dir: Path to the root results directory.

    Returns:
        Dict mapping condition names to lists of result dictionaries (one per seed).
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Directory {results_dir} not found.")

    all_results = {}

    for seed_dir in results_dir.iterdir():
        if not seed_dir.is_dir():
            continue

        for condition_dir in seed_dir.iterdir():
            if not condition_dir.is_dir():
                continue

            condition = condition_dir.name
            results_path = condition_dir / "results.json"

            if not results_path.exists():
                continue

            try:
                with open(results_path) as f:
                    data = json.load(f)

                if condition not in all_results:
                    all_results[condition] = []

                # Deduplicate seeds in case of multiple runs with same seed in same dir (unlikely but safe)
                # Usually we just append
                all_results[condition].append(data)

            except json.JSONDecodeError:
                print(f"Warning: Corrupted JSON file {results_path}")
            except Exception as e:
                print(f"Warning: Failed to load {results_path}: {e}")

    return all_results
