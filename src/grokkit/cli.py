import argparse
import json
from pathlib import Path
from typing import List, Optional
import sys

from .parser import collect_runs, parse_run_results
from .figures import aggregate_runs, generate_markdown_table, plot_training_trajectory
from .cliff import find_cliff

def do_analyze(args: argparse.Namespace):
    run_dir = Path(args.run_dir)
    runs = collect_runs(run_dir)

    if not runs:
        results_path = run_dir / "results.json"
        if results_path.exists():
            data = parse_run_results(results_path)
            if data:
                data["condition"] = run_dir.name
                runs = [data]

    if not runs:
        print(f"No valid results found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    summary = aggregate_runs(runs)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"# Analysis for {run_dir}\n")
        print(generate_markdown_table(summary))

    if args.plot:
        out_path = Path("training_trajectories.png")
        plot_training_trajectory(runs, out_path)
        print(f"\nSaved plot to {out_path}")

def do_compare(args: argparse.Namespace):
    all_runs = []
    for d in args.dirs:
        p = Path(d)
        runs = collect_runs(p)
        if not runs:
            results_path = p / "results.json"
            if results_path.exists():
                data = parse_run_results(results_path)
                if data:
                    data["condition"] = p.name
                    runs = [data]
        all_runs.extend(runs)

    if not all_runs:
        print("No valid results found in any directories", file=sys.stderr)
        sys.exit(1)

    summary = aggregate_runs(all_runs)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("# Cross-condition Comparison\n")
        print(generate_markdown_table(summary))

def do_cliff(args: argparse.Namespace):
    run_dir = Path(args.run_dir)
    runs = collect_runs(run_dir)

    if not runs:
        print(f"No valid results found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    cliff = find_cliff(runs, x_key=args.x_key, y_key=args.y_key, value_key=args.value_key, threshold=args.threshold)

    if args.json:
        print(json.dumps(cliff, indent=2))
    else:
        print(f"# Cliff detection ({args.x_key} -> min {args.y_key} below {args.threshold})\n")
        for x_val, y_val in cliff.items():
            print(f"- {args.x_key}={x_val}: {args.y_key}={y_val}")

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Grokkit: Analysis package for grokking-collapse")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_analyze = subparsers.add_parser("analyze", help="Full standard report for a run directory")
    p_analyze.add_argument("run_dir", help="Directory containing condition subdirectories or a results.json")
    p_analyze.add_argument("--json", action="store_true", help="Output JSON instead of markdown")
    p_analyze.add_argument("--plot", action="store_true", help="Generate trajectory plots")
    p_analyze.set_defaults(func=do_analyze)

    p_compare = subparsers.add_parser("compare", help="Cross-condition comparison tables")
    p_compare.add_argument("dirs", nargs="+", help="Directories to compare")
    p_compare.add_argument("--json", action="store_true", help="Output JSON instead of markdown")
    p_compare.set_defaults(func=do_compare)

    p_cliff = subparsers.add_parser("cliff", help="Cliff stats with CIs")
    p_cliff.add_argument("run_dir", help="Directory containing grid search results")
    p_cliff.add_argument("--x-key", default="wd", help="X-axis key for cliff detection")
    p_cliff.add_argument("--y-key", default="noise", help="Y-axis key for cliff detection")
    p_cliff.add_argument("--value-key", default="final_fourier_concentration", help="Value key to check against threshold")
    p_cliff.add_argument("--threshold", type=float, default=0.20, help="Threshold value")
    p_cliff.add_argument("--json", action="store_true", help="Output JSON instead of markdown")
    p_cliff.set_defaults(func=do_cliff)

    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
