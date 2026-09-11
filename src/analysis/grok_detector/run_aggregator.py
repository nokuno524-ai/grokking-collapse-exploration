import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .detectors import piecewise_linear_detector, bootstrap_ci

def load_run_history(run_dir: Path) -> List[Dict]:
    """Load the history array from a results.json file in a given directory."""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return []
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
            return data.get("history", [])
    except Exception:
        return []

def extract_metric(history: List[Dict], metric: str = "test_acc") -> Tuple[np.ndarray, np.ndarray]:
    """Extract steps and valid metric values, filtering out None or NaN."""
    steps = []
    values = []
    for entry in history:
        val = entry.get(metric)
        if val is not None and not np.isnan(val):
            steps.append(entry.get("step", 0))
            values.append(val)
    return np.array(steps, dtype=float), np.array(values, dtype=float)

def aggregate_condition_runs(condition_dir: Path, metric: str = "test_acc",
                             detector_fn=piecewise_linear_detector,
                             calc_ci: bool = True) -> Dict[str, Any]:
    """
    Ingest multiple seed runs. Note that for this project, the structure is
    results/multi_seed/<seed>/<condition>/results.json. So condition_dir here
    is passed in as results/multi_seed/<condition>, which is NOT the actual path.
    We need to scan the parent directory for seed folders.
    """
    run_results = []
    grok_steps = []

    # Actually results_dir is the parent of condition_dir
    results_dir = condition_dir.parent
    cond_name = condition_dir.name

    for seed_dir in [d for d in results_dir.iterdir() if d.is_dir()]:
        actual_cond_dir = seed_dir / cond_name
        history = load_run_history(actual_cond_dir)
        if not history:
            continue

        steps, values = extract_metric(history, metric)
        if len(steps) < 4:
            continue

        if calc_ci:
            cp, lower, upper = bootstrap_ci(steps, values, detector_fn)
        else:
            cp = detector_fn(steps, values)
            lower, upper = cp, cp

        is_grokked = False
        final_val = values[-1] if len(values) > 0 else 0
        if cp is not None and final_val >= 0.95 - 1e-6:
            is_grokked = True
            grok_steps.append(cp)

        run_results.append({
            "seed": seed_dir.name,
            "grokked": is_grokked,
            "grok_step": cp,
            "grok_step_ci": (lower, upper) if cp is not None else None,
            "max_step": float(steps[-1]) if len(steps) > 0 else None,
            "final_value": final_val
        })

    stats = {}
    if grok_steps:
        stats["median"] = float(np.median(grok_steps))
        q25, q75 = np.percentile(grok_steps, [25, 75])
        stats["iqr"] = float(q75 - q25)
        stats["q25"] = float(q25)
        stats["q75"] = float(q75)
        stats["max_grok_step"] = float(np.max(grok_steps))
    else:
        stats["median"] = None
        stats["iqr"] = None
        stats["q25"] = None
        stats["q75"] = None
        stats["max_grok_step"] = None

    stats["num_runs"] = len(run_results)
    stats["num_grokked"] = sum(1 for r in run_results if r["grokked"])

    return {
        "condition": cond_name,
        "runs": run_results,
        "stats": stats
    }
