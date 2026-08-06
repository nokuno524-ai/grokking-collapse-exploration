import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_experiment(script_path, args=None):
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Reproduce all key experiments.")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode (few steps)")
    parser.add_argument("--output-dir", type=str, default="results/reproduce", help="Output directory")
    args = parser.parse_args()

    max_steps = "100" if args.quick else "50000"

    # 1. Run all conditions from train.py
    print("--- Running main conditions ---")
    run_experiment("src/train.py", ["--all", "--max-steps", max_steps, "--output-dir", args.output_dir])

    # 2. Run baselines
    print("--- Running noise baseline ---")
    run_experiment("src/run_noise_baseline.py", ["--max-steps", max_steps, "--output-dir", os.path.join(args.output_dir, "noise_baseline")])

    print("--- Running scarcity baseline ---")
    run_experiment("src/run_scarcity_baseline.py", ["--max-steps", max_steps, "--output-dir", os.path.join(args.output_dir, "scarcity_baseline")])

    print(f"All experiments finished. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
