"""
Runner that executes the multi-seed sweep and aggregates stats.
"""
import argparse
import multiprocessing
import sys
import json
from pathlib import Path

# Add src to Python path if necessary
sys.path.append(str(Path(__file__).parent))

try:
    from run_multi_seed import main as multi_seed_main, run_one, build_tasks
    from analysis.grok_detector.run_aggregator import aggregate_runs
    from analysis.grok_detector.stats import kaplan_meier_estimator, hedges_g, permutation_test
except ImportError:
    from src.run_multi_seed import main as multi_seed_main, run_one, build_tasks
    from src.analysis.grok_detector.run_aggregator import aggregate_runs
    from src.analysis.grok_detector.stats import kaplan_meier_estimator, hedges_g, permutation_test

import numpy as np

def run_task_wrapper(args):
    seed, cond_name, cond_cfg, output_root, max_steps = args
    return run_one(seed, cond_name, cond_cfg, output_root, max_steps)

def generate_multi_seed_stats(results_dir: str, max_steps: int):
    results_path = Path(results_dir)
    agg = aggregate_runs(results_path, max_steps=max_steps)

    # Save a tidy JSON summary and JSONL for survival
    tidy_records = []

    for cond_name, runs in agg.items():
        if not runs:
            continue

        durations = np.array([r["time_to_grok"] for r in runs])
        observed = np.array([r["event_observed"] for r in runs])

        # Survival analysis KM estimator
        km = kaplan_meier_estimator(durations, observed)

        print(f"\nCondition: {cond_name}")
        print(f"  Runs: {len(runs)}")
        print(f"  Grokked: {np.sum(observed)}/{len(runs)}")
        print(f"  Median grok step: {np.median(durations[observed]) if np.sum(observed) > 0 else '>'+str(max_steps)}")

        for r in runs:
            tidy_records.append({
                "condition": cond_name,
                "seed": r["seed"],
                "grokked": r["grokked"],
                "grokking_step": r["grokking_step"],
                "time_to_grok": r["time_to_grok"],
                "event_observed": r["event_observed"],
                "final_test_acc": r["final_test_acc"]
            })

    summary_file = results_path / "stats_summary.jsonl"
    with open(summary_file, 'w') as f:
        for rec in tidy_records:
            f.write(json.dumps(rec, sort_keys=True) + '\n')

    print(f"\nStats summary saved to {summary_file}")

    # Effect sizes vs pure
    if "pure" in agg and len(agg["pure"]) > 1:
        pure_durations = np.array([r["time_to_grok"] for r in agg["pure"]])
        print("\nEffect Sizes (vs pure):")
        for cond_name, runs in agg.items():
            if cond_name == "pure" or not runs: continue

            cond_durations = np.array([r["time_to_grok"] for r in runs])
            g, ci_low, ci_high = hedges_g(pure_durations, cond_durations)
            pval = permutation_test(pure_durations, cond_durations)

            print(f"  {cond_name}:")
            print(f"    Hedge's g: {g:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
            print(f"    Permutation p-value: {pval:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="results/multi_seed")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--aggregate-only", action="store_true", help="Only run aggregation on existing results")
    args = parser.parse_args()

    if not args.aggregate_only:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        tasks = build_tasks(seeds)
        print(f"Total tasks: {len(tasks)} ({len(seeds)} seeds)")

        run_args = [(t[0], t[1], t[2], args.output_dir, args.max_steps) for t in tasks]

        if args.jobs > 1:
            with multiprocessing.Pool(args.jobs) as pool:
                pool.map(run_task_wrapper, run_args)
        else:
            for ra in run_args:
                run_task_wrapper(ra)

    generate_multi_seed_stats(args.output_dir, args.max_steps)

if __name__ == "__main__":
    main()
