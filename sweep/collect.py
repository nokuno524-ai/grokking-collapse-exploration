"""
Collects and aggregates results from completed sweep runs.
Scans through generated experiment directories (or reads from a manifest),
reads the individual results.json files, and outputs a combined CSV.
Tracks status as COMPLETE, INCOMPLETE, or MISSING.
"""

import argparse
import json
import csv
import os
from pathlib import Path

def parse_args():
    """Parse command-line arguments for sweep collection."""
    parser = argparse.ArgumentParser(description="Collect sweep results from generated directories.")
    parser.add_argument("--run-dir", required=True, help="Directory containing the run subdirectories or manifest.csv")
    parser.add_argument("--out-csv", required=True, help="Output CSV path for aggregated results")
    return parser.parse_args()

def collect_results():
    """
    Main logic to iterate over sweep runs, extract final metrics and configs,
    determine the run status, and write all records to an aggregated CSV file.
    """
    args = parse_args()
    run_dir = Path(args.run_dir)

    manifest_path = run_dir / "manifest.csv"

    runs = []
    param_keys = []

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            param_keys = [k for k in reader.fieldnames if k not in ("run_dir", "exp_name", "seed")]
            for row in reader:
                runs.append({
                    "run_dir": row["run_dir"],
                    "exp_name": row["exp_name"],
                    "seed": int(row["seed"]),
                    "params": {k: row[k] for k in param_keys}
                })
    else:
        # Fallback if no manifest exists, scan directories
        for p in run_dir.iterdir():
            if p.is_dir() and (p / "config.json").exists():
                with open(p / "config.json", "r") as f:
                    cfg = json.load(f)

                # Try to guess params based on the directory name or just save all standard config keys we care about
                # For this fallback, we'll just track seed, and try to extract from name
                name_parts = p.name.split("_")
                seed = cfg.get("seed", 42)
                exp_name = cfg.get("condition_name", p.name)

                # Try to extract key=value pairs
                params = {}
                for part in name_parts:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        params[k] = v
                        if k not in param_keys:
                            param_keys.append(k)

                runs.append({
                    "run_dir": str(p),
                    "exp_name": exp_name,
                    "seed": seed,
                    "params": params
                })

    out_rows = []

    for run in runs:
        d = Path(run["run_dir"])
        results_file = d / "results.json"

        row = {
            "run_dir": str(d),
            "exp_name": run["exp_name"],
            "seed": run["seed"],
        }
        for k in param_keys:
            row[k] = run["params"].get(k, "")

        if not results_file.exists():
            row["status"] = "MISSING"
            row["grok_step"] = float('nan')
            row["final_acc"] = float('nan')
            row["final_weight_norm"] = float('nan')
        else:
            try:
                with open(results_file, "r") as f:
                    res = json.load(f)

                row["grok_step"] = res.get("grokking_step") if res.get("grokking_step") is not None else float('nan')
                row["final_acc"] = res.get("final_test_acc", float('nan'))
                row["final_weight_norm"] = res.get("final_weight_norm", float('nan'))

                # Decide complete/incomplete
                # Complete if final_test_acc is present and history reaches max_steps? Or just valid JSON
                row["status"] = "COMPLETE"

            except json.JSONDecodeError:
                row["status"] = "INCOMPLETE"
                row["grok_step"] = float('nan')
                row["final_acc"] = float('nan')
                row["final_weight_norm"] = float('nan')

        out_rows.append(row)

    if not out_rows:
        print("No runs found.")
        return

    # Write aggregated CSV
    with open(args.out_csv, "w", newline="") as f:
        # Define fieldnames
        fieldnames = ["run_dir", "exp_name", "status"] + param_keys + ["seed", "grok_step", "final_acc", "final_weight_norm"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"Collected results from {len(out_rows)} runs into {args.out_csv}")

if __name__ == "__main__":
    collect_results()
