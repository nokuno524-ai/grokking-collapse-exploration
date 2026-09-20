"""
Ensemble sweep runner.
Generates a configuration matrix of conditions x seeds and supports
both local sequential execution and generating sbatch scripts for SLURM.
Includes resumability: skips tasks where results.json already exists.
"""

import argparse
import sys
from pathlib import Path

# Add src to python path so we can import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.train import TrainConfig, train
    from src.data import get_all_conditions
except ImportError:
    from train import TrainConfig, train
    from data import get_all_conditions

def build_config_matrix(seeds: list[int], condition_names: list[str]) -> list[dict]:
    all_conditions = get_all_conditions()
    tasks = []

    # Filter conditions if specific ones are requested, else use all
    if condition_names:
        conds = {name: all_conditions[name] for name in condition_names if name in all_conditions}
    else:
        conds = all_conditions

    for seed in seeds:
        for name, config in conds.items():
            tasks.append({
                "seed": seed,
                "condition_name": name,
                "config": config
            })
    return tasks

def is_task_completed(output_dir: Path) -> bool:
    """Check if the results.json file exists for this task."""
    return (output_dir / "results.json").exists()

def run_task_locally(task: dict, output_root: Path, max_steps: int):
    """Run a single task using the train function."""
    seed = task["seed"]
    cond_name = task["condition_name"]
    cond_cfg = task["config"]

    output_dir = output_root / str(seed)
    # The train function appends condition_name to output_dir, so the final path is output_dir / cond_name
    expected_final_dir = output_dir / cond_name

    if is_task_completed(expected_final_dir):
        print(f"Skipping completed task: seed={seed}, condition={cond_name}")
        return

    print(f"Running task: seed={seed}, condition={cond_name}")

    train_config = TrainConfig(
        prime=cond_cfg.prime,
        train_fraction=0.3,
        lr=1e-3,
        weight_decay=1.0,
        collapse_level=cond_cfg.collapse_level,
        collapse_severity=cond_cfg.collapse_severity,
        seed=seed,
        condition_name=cond_name,
        output_dir=str(output_dir),
        max_steps=max_steps,
    )
    train(train_config)

def generate_sbatch_script(tasks: list[dict], output_root: Path, max_steps: int, script_path: Path):
    """Generate a SLURM sbatch script to run the matrix as an array job."""

    # Write a wrapper script that takes an array ID and runs the task
    wrapper_path = script_path.parent / f"{script_path.stem}_wrapper.py"
    with open(wrapper_path, "w") as f:
        f.write(f'''#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ensemble.runner import build_config_matrix, run_task_locally

seeds = {[t["seed"] for t in tasks]}
# De-duplicate maintaining order
seeds = list(dict.fromkeys(seeds))
condition_names = {[t["condition_name"] for t in tasks]}
condition_names = list(dict.fromkeys(condition_names))

tasks = build_config_matrix(seeds, condition_names)

if len(sys.argv) != 2:
    print("Usage: wrapper.py <array_id>")
    sys.exit(1)

array_id = int(sys.argv[1])
if array_id < 0 or array_id >= len(tasks):
    print(f"Array ID {{array_id}} out of bounds")
    sys.exit(1)

task = tasks[array_id]
run_task_locally(task, Path("{output_root}"), {max_steps})
''')

    # Write the sbatch script
    with open(script_path, "w") as f:
        f.write(f'''#!/bin/bash
#SBATCH --job-name=ensemble_sweep
#SBATCH --output={output_root.parent}/logs/ensemble_%A_%a.out
#SBATCH --error={output_root.parent}/logs/ensemble_%A_%a.err
#SBATCH --array=0-{len(tasks)-1}
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Ensure logs directory exists
mkdir -p {output_root.parent}/logs

echo "Starting task $SLURM_ARRAY_TASK_ID"
python3 {wrapper_path} $SLURM_ARRAY_TASK_ID
echo "Finished task $SLURM_ARRAY_TASK_ID"
''')

    print(f"Generated SLURM scripts at: \n  {script_path}\n  {wrapper_path}")
    print(f"To run: sbatch {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Ensemble Sweep Runner")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated list of seeds")
    parser.add_argument("--conditions", type=str, default="", help="Comma-separated list of conditions (leave empty for all)")
    parser.add_argument("--output-dir", type=str, default="results/ensemble_sweep", help="Output directory root")
    parser.add_argument("--max-steps", type=int, default=50000, help="Max training steps per run")
    parser.add_argument("--mode", type=str, choices=["local", "sbatch"], default="local", help="Execution mode")
    parser.add_argument("--sbatch-file", type=str, default="slurm/ensemble_sweep.sbatch", help="Path for generated sbatch script")

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    output_root = Path(args.output_dir)

    tasks = build_config_matrix(seeds, conditions)
    print(f"Generated {len(tasks)} tasks.")

    if args.mode == "local":
        for i, task in enumerate(tasks):
            print(f"--- Task {i+1}/{len(tasks)} ---")
            run_task_locally(task, output_root, args.max_steps)
    elif args.mode == "sbatch":
        sbatch_path = Path(args.sbatch_file)
        sbatch_path.parent.mkdir(parents=True, exist_ok=True)
        generate_sbatch_script(tasks, output_root, args.max_steps, sbatch_path)

if __name__ == "__main__":
    main()
