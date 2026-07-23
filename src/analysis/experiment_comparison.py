import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

def load_results(base_dir: str) -> Dict[str, List[dict]]:
    """Loads results from multiple runs and returns a dictionary of conditions to lists of run results."""
    base_path = Path(base_dir)
    results = {}
    for condition_dir in base_path.iterdir():
        if not condition_dir.is_dir():
            continue
        condition = condition_dir.name
        results[condition] = []

        # Check if it has seed subdirectories or direct results
        seed_dirs = [d for d in condition_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]

        if seed_dirs:
            for seed_dir in seed_dirs:
                results_file = seed_dir / "results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results[condition].append(json.load(f))
        else:
            results_file = condition_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results[condition].append(json.load(f))
    return results

def compute_bootstrap_ci(data: List[float], n_resamples: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Computes the mean and bootstrap confidence interval for a list of values."""
    if not data:
        return np.nan, np.nan, np.nan

    data = np.array(data)
    n = len(data)
    mean = np.mean(data)

    if n < 2:
        return mean, mean, mean

    resamples = np.random.choice(data, size=(n_resamples, n), replace=True)
    means = np.mean(resamples, axis=1)
    lower = np.percentile(means, alpha / 2 * 100)
    upper = np.percentile(means, (1 - alpha / 2) * 100)

    return mean, lower, upper
