import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional

def parse_run_results(results_path: Path) -> Optional[Dict[str, Any]]:
    """
    Safely reads a results.json file, fixing common logging issues like NaNs
    and duplicate evaluation records.
    """
    if not results_path.exists():
        return None
    try:
        with open(results_path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None

    history = data.get("history", [])
    clean_history = []
    seen_steps = set()

    for entry in history:
        step = entry.get("step")
        if step is None or step in seen_steps:
            continue
        seen_steps.add(step)

        clean_entry = {}
        for k, v in entry.items():
            if isinstance(v, float) and math.isnan(v):
                clean_entry[k] = float('nan')
            elif isinstance(v, str) and v.lower() == 'nan':
                clean_entry[k] = float('nan')
            else:
                clean_entry[k] = v
        clean_history.append(clean_entry)

    data["history"] = clean_history
    return data

def collect_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Collects multiple run results from subdirectories.
    Traverses recursively to support nested grids.
    """
    runs = []
    if not results_dir.exists() or not results_dir.is_dir():
        return runs

    for results_path in results_dir.rglob("results.json"):
        data = parse_run_results(results_path)
        if data is not None:
            # Flatten config into data for easy access if present
            if "config" in data:
                # also ensure keys are top level
                for k, v in data["config"].items():
                    data[k] = v
                for k, v in data["config"].items():
                    if k not in data:
                        data[k] = v
            # Set a generic condition string if one isn't explicitly defined
            if "condition" not in data:
                # If it's a simple flat struct, use parent dir name. Otherwise, try to infer from config.
                # A good fallback is the relative path
                rel = results_path.parent.relative_to(results_dir)
                data["condition"] = str(rel) if str(rel) != "." else results_path.parent.name
            runs.append(data)

    return runs
