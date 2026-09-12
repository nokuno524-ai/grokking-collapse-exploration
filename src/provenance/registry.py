import json
import os
from typing import Dict, Any, List

REGISTRY_PATH = "experiments/registry.json"

def _load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def _save_registry(data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def register(run_id: str, entry: Dict[str, Any]) -> None:
    """Register a new experiment run."""
    registry = _load_registry()
    if run_id in registry:
        raise ValueError(f"Run ID {run_id} already exists in registry.")
    registry[run_id] = entry
    _save_registry(registry)

def query(filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Query experiments matching filters."""
    registry = _load_registry()
    if not filters:
        return list(registry.values())

    results = []
    for entry in registry.values():
        match = True
        for k, v in filters.items():
            if entry.get(k) != v:
                match = False
                break
        if match:
            results.append(entry)
    return results

def diff(run_id_1: str, run_id_2: str) -> Dict[str, Any]:
    """Diff two registered experiments."""
    registry = _load_registry()
    if run_id_1 not in registry:
        raise ValueError(f"Run ID {run_id_1} not found.")
    if run_id_2 not in registry:
        raise ValueError(f"Run ID {run_id_2} not found.")

    entry_1 = registry[run_id_1]
    entry_2 = registry[run_id_2]

    # Simple diffing logic
    keys = set(entry_1.keys()).union(entry_2.keys())
    differences = {}
    for k in keys:
        if entry_1.get(k) != entry_2.get(k):
            differences[k] = {
                "run_1": entry_1.get(k),
                "run_2": entry_2.get(k)
            }
    return differences

def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment Registry CLI")
    subparsers = parser.add_subparsers(dest="command")

    # list/query command
    query_parser = subparsers.add_parser("query")
    # allow some filters
    query_parser.add_argument("--run-id", help="Filter by run id")

    diff_parser = subparsers.add_parser("diff")
    diff_parser.add_argument("run_id_1")
    diff_parser.add_argument("run_id_2")

    args = parser.parse_args()

    if args.command == "query":
        filters = {}
        if args.run_id:
            filters["run_id"] = args.run_id
        res = query(filters)
        print(json.dumps(res, indent=2))
    elif args.command == "diff":
        try:
            res = diff(args.run_id_1, args.run_id_2)
            print(json.dumps(res, indent=2))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    cli()
