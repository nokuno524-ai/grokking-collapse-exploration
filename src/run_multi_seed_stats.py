"""
Multi-seed sweep across the 5 collapse conditions with multiprocessing and statistical aggregation.

Layout:
  results/multi_seed/<seed>/<condition>/results.json

Outputs:
  Aggregated metrics CSV and JSON containing summary statistics.
"""

import argparse
import concurrent.futures
from pathlib import Path
import json

try:
    from .train import TrainConfig, train
    from .data import get_all_conditions
    from .analysis.grok_detector.run_aggregator import aggregate_and_plot
except ImportError:
    from train import TrainConfig, train
    from data import get_all_conditions
    from analysis.grok_detector.run_aggregator import aggregate_and_plot


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
CONDITION_ORDER = ["pure", "low_collapse", "medium_collapse", "high_collapse", "severe_collapse"]


def build_tasks(seeds):
    conditions = get_all_conditions()
    tasks = []
    for seed in seeds:
        for cond_name in CONDITION_ORDER:
            tasks.append((seed, cond_name, conditions[cond_name]))
    return tasks


def run_one(seed: int, cond_name: str, cond_cfg, output_root: str, max_steps: int):
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
    # We suppress output if running concurrently to avoid log interleaving,
    # but actual stdout logic depends on train.py. We'll just return the result.
    return train(train_config)


def run_task_wrapper(args):
    seed, cond_name, cond_cfg, output_root, max_steps = args
    print(f"Starting seed={seed} condition={cond_name}...")
    try:
        res = run_one(seed, cond_name, cond_cfg, output_root, max_steps)
        print(f"Finished seed={seed} condition={cond_name}")
        return res
    except Exception as e:
        print(f"Error in seed={seed} condition={cond_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated list of seeds (default: 42,43,44,45,46)")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/multi_seed")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel processes to use.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip training and just run aggregation on existing results.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = build_tasks(seeds)
    print(f"Total tasks: {len(tasks)} ({len(seeds)} seeds × {len(CONDITION_ORDER)} conditions)")

    if not args.aggregate_only:
        task_args = [(seed, cond_name, cond_cfg, args.output_dir, args.max_steps) for seed, cond_name, cond_cfg in tasks]

        if args.workers > 1:
            print(f"Running with {args.workers} workers...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(run_task_wrapper, task_args))
        else:
            for i, task_arg in enumerate(task_args):
                print(f"\n{'='*60}\n[{i+1}/{len(tasks)}] seed={task_arg[0]} condition={task_arg[1]}\n{'='*60}")
                run_task_wrapper(task_arg)

    print("\nRunning statistical aggregation...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = str(Path(args.output_dir) / "results_aggregated.csv")
    json_path = str(Path(args.output_dir) / "stats_summary.json")

    stats = aggregate_and_plot(args.output_dir, output_csv=csv_path)

    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(f"Aggregation complete. Results saved to {csv_path} and {json_path}.")


if __name__ == "__main__":
    main()
