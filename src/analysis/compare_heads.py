import csv
import argparse
from pathlib import Path
from typing import Dict, List

def load_importance_csv(filepath: Path) -> Dict[str, float]:
    """
    Load a head importance CSV and return a dictionary mapping
    'layer_L_head_H' to the accuracy drop.
    """
    importances = {}
    with filepath.open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"layer_{row['layer']}_head_{row['head']}"
            importances[key] = float(row['acc_drop'])
    return importances

def compare_head_importances(pure_csv: Path, collapsed_csv: Path, output_csv: Path = None):
    """
    Compare head importance scores between a pure run and a collapsed run.
    Outputs a new CSV with the comparison if output_csv is provided.
    """
    pure_importances = load_importance_csv(pure_csv)
    collapsed_importances = load_importance_csv(collapsed_csv)

    results = []

    # We assume both CSVs have the same heads
    for key in pure_importances.keys():
        if key in collapsed_importances:
            pure_drop = pure_importances[key]
            collapsed_drop = collapsed_importances[key]
            diff = pure_drop - collapsed_drop

            results.append({
                "head_id": key,
                "pure_acc_drop": pure_drop,
                "collapsed_acc_drop": collapsed_drop,
                "difference": diff,
            })

    # Sort by the largest difference (heads most important for pure that are less important for collapsed)
    results.sort(key=lambda x: x["difference"], reverse=True)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["head_id", "pure_acc_drop", "collapsed_acc_drop", "difference"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Comparison saved to {output_csv}")

    # Print the top differences
    print(f"{'Head':<20} | {'Pure Drop':<12} | {'Collapsed Drop':<15} | {'Difference'}")
    print("-" * 65)
    for res in results:
        print(f"{res['head_id']:<20} | {res['pure_acc_drop']:<12.4f} | {res['collapsed_acc_drop']:<15.4f} | {res['difference']:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Compare head importances between two runs.")
    parser.add_argument("--pure-csv", type=Path, required=True, help="Path to the pure run head importance CSV.")
    parser.add_argument("--collapsed-csv", type=Path, required=True, help="Path to the collapsed run head importance CSV.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the comparison CSV.")

    args = parser.parse_args()

    compare_head_importances(args.pure_csv, args.collapsed_csv, args.output)

if __name__ == "__main__":
    main()
