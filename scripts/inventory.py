"""
Inventory script to iterate over all results.json outputs and extract
relevant metrics (accuracy, grokking step, weight norm) to a CSV summary.
"""
import json
import glob
import os
import csv
from pathlib import Path

def main():
    results_dir = Path("results")
    rows = []

    # We will search for all results.json files in results directory
    for path in results_dir.rglob("results.json"):
        try:
            with open(path) as f:
                data = json.load(f)

            config = data.get("config", {})
            collapse_ratio = config.get("collapse_level", 0.0)
            collapse_severity = config.get("collapse_severity", 0.0)
            seed = config.get("seed", 42)
            condition = config.get("condition_name", path.parent.name)

            # Grokking step definition from memory: first step where val acc > 90%
            history = data.get("history", [])
            grokking_step = data.get("grokking_step")

            final_acc = data.get("final_test_acc")
            if final_acc is None and history:
                final_acc = history[-1].get("test_acc")

            final_weight_norm = data.get("final_weight_norm")
            if final_weight_norm is None and history:
                final_weight_norm = history[-1].get("weight_norm")

            if grokking_step is None:
                for entry in history:
                    if entry.get("test_acc", 0) > 0.9:
                        grokking_step = entry.get("step")
                        break

            # Use path relative to results as an identifier if condition is just "seed_42"
            run_id = str(path.parent.relative_to(results_dir))

            rows.append({
                "run_id": run_id,
                "condition": condition,
                "collapse_ratio": collapse_ratio,
                "collapse_severity": collapse_severity,
                "seed": seed,
                "final_accuracy": final_acc,
                "grokking_step": grokking_step,
                "final_weight_norm": final_weight_norm
            })
        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Save to results/summary.csv
    output_path = Path("results/summary.csv")
    with open(output_path, "w", newline="") as f:
        fieldnames = ["run_id", "condition", "collapse_ratio", "collapse_severity", "seed", "final_accuracy", "grokking_step", "final_weight_norm"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Inventory saved to {output_path} with {len(rows)} entries.")

if __name__ == "__main__":
    main()
