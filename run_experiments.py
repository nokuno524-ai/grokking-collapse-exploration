import argparse
import subprocess
import os
import json
from pathlib import Path
import sys

CONDITIONS = ['pure', 'low', 'medium', 'high', 'severe']

def check_run_completed(output_dir: str, condition: str) -> bool:
    """Check if a condition has already been fully run."""
    results_path = Path(output_dir) / condition / "results.json"
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
            # Check if grokked key exists, which means the run finished
            if "grokked" in data:
                return True
        except Exception:
            pass
    return False

def run_experiment(condition: str, max_steps: int, output_dir: str):
    """Run a single experiment condition via subprocess to isolate memory/GPU."""
    print(f"\n{'='*60}")
    print(f"Running condition: {condition}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "src/train.py",
        "--condition", condition,
        "--max-steps", str(max_steps),
        "--output-dir", output_dir
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running condition {condition}: {e}")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Unified Experiment Runner for Model Collapse vs Grokking")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS,
                        help="List of conditions to run (default: all)")
    parser.add_argument("--max-steps", type=int, default=50000,
                        help="Maximum training steps per run")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if results exist")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    summary = {}

    for condition in args.conditions:
        if condition not in CONDITIONS:
            print(f"Warning: Unknown condition '{condition}'. Skipping.")
            continue

        if not args.force and check_run_completed(args.output_dir, condition):
            print(f"Skipping {condition}: already completed. (Use --force to override)")
            # Load existing results for summary
            results_path = Path(args.output_dir) / condition / "results.json"
            with open(results_path, 'r') as f:
                summary[condition] = json.load(f)
            continue

        success = run_experiment(condition, args.max_steps, args.output_dir)

        if success:
            results_path = Path(args.output_dir) / condition / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    summary[condition] = json.load(f)

    # Print Final Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for name, r in summary.items():
        if "grokked" in r:
            status = "✅ GROKKED" if r["grokked"] else "❌ NO GROK"
            step = r.get('grokking_step', 'N/A')
            test_acc = r.get('final_test_acc', 0.0)
            fourier = r.get('final_fourier_concentration', 0.0)
            print(f"  {name:20s} | {status} | step={step} | "
                  f"test_acc={test_acc:.4f} | fourier={fourier:.3f}")

if __name__ == "__main__":
    main()
