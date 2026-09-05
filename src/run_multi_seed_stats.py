"""
Multi-seed runner for statistical robustness analysis of model collapse.
Collects full accuracy trajectories and weight-norm trajectories,
and outputs results as tidy CSV / .jsonl files for survival analysis.
"""

import argparse
import json
import csv
from pathlib import Path
import multiprocessing
from typing import List, Dict, Any

try:
    from src.train import TrainConfig, train
    from src.data import get_all_conditions
except ImportError:
    from train import TrainConfig, train
    from data import get_all_conditions


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
CONDITION_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]


def build_tasks(seeds: List[int]) -> List[tuple]:
    """Build a list of tasks for the multi-seed experiment."""
    conditions = get_all_conditions()
    tasks = []
    for seed in seeds:
        for cond_name in CONDITION_ORDER:
            tasks.append((seed, cond_name, conditions[cond_name]))
    return tasks


def run_one(args: tuple) -> Dict[str, Any]:
    """Run a single experiment."""
    seed, cond_name, cond_cfg, output_root, max_steps = args

    train_config = TrainConfig(
        prime=cond_cfg.prime,
        train_fraction=0.3,
        lr=1e-3,
        weight_decay=1.0,
        collapse_level=cond_cfg.collapse_level,
        collapse_severity=cond_cfg.collapse_severity,
        seed=seed,
        condition_name=cond_name,
        output_dir=str(Path(output_root) / str(seed)),
        max_steps=max_steps,
    )

    state = train(train_config)

    # Return serializable result
    return {
        "seed": seed,
        "condition": cond_name,
        "collapse_level": cond_cfg.collapse_level,
        "collapse_severity": cond_cfg.collapse_severity,
        "grokked": state.grokked,
        "grokking_step": state.grokking_step,
        "history": state.history
    }


def save_results(results: List[Dict[str, Any]], output_dir: str):
    """Save results as JSONL and CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save as JSONL for detailed history/survival analysis
    jsonl_path = out_path / "results.jsonl"
    with open(jsonl_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    # Save a tidy CSV for summary analysis
    csv_path = out_path / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["seed", "condition", "collapse_level", "collapse_severity", "step", "train_loss", "test_loss", "train_acc", "test_acc", "weight_norm"])

        for res in results:
            seed = res["seed"]
            cond = res["condition"]
            level = res["collapse_level"]
            severity = res["collapse_severity"]

            for entry in res["history"]:
                writer.writerow([
                    seed, cond, level, severity,
                    entry.get("step"),
                    entry.get("train_loss"),
                    entry.get("test_loss"),
                    entry.get("train_acc"),
                    entry.get("test_acc"),
                    entry.get("weight_norm")
                ])

    print(f"\nSaved results to {jsonl_path} and {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed experiments for statistical analysis.")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated list of seeds (default: 42,43,44,45,46)")
    parser.add_argument("--max-steps", type=int, default=50000, help="Maximum number of training steps per run.")
    parser.add_argument("--output-dir", type=str, default="results/multi_seed", help="Directory to save overall results.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs to run (default: 1).")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = build_tasks(seeds)
    print(f"Total tasks: {len(tasks)} ({len(seeds)} seeds × {len(CONDITION_ORDER)} conditions)")

    # Prepare arguments for multiprocessing
    run_args = [(seed, cond_name, cond_cfg, args.output_dir, args.max_steps) for seed, cond_name, cond_cfg in tasks]

    results = []
    if args.jobs > 1:
        with multiprocessing.Pool(processes=args.jobs) as pool:
            for i, res in enumerate(pool.imap_unordered(run_one, run_args)):
                print(f"[{i+1}/{len(tasks)}] Finished seed={res['seed']} condition={res['condition']}")
                results.append(res)
    else:
        for i, task_args in enumerate(run_args):
            print(f"\n{'='*60}\n[{i+1}/{len(tasks)}] seed={task_args[0]} condition={task_args[1]}\n{'='*60}")
            res = run_one(task_args)
            results.append(res)

    save_results(results, args.output_dir)

if __name__ == "__main__":
    main()
