import json
import argparse
from pathlib import Path

def detect_grokking_onset(results_dir="results"):
    run_paths = [p for p in Path(results_dir).iterdir() if p.is_dir()]

    print(f"{'Condition':<20} | {'Grokking Step':<15} | {'Final Train Acc':<15} | {'Final Test Acc':<15}")
    print("-" * 70)

    for run_path in run_paths:
        res_file = run_path / "results.json"
        if not res_file.exists():
            continue

        with open(res_file, "r") as f:
            data = json.load(f)

        grokked = data.get("grokked", False)
        step = data.get("grokking_step", "N/A") if grokked else "Did not grok"

        train_acc = data.get("final_train_acc", 0.0)
        test_acc = data.get("final_test_acc", 0.0)

        print(f"{run_path.name:<20} | {str(step):<15} | {train_acc:<15.4f} | {test_acc:<15.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    detect_grokking_onset(args.results_dir)
