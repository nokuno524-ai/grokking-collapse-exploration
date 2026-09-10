"""
Difficulty sweep config generator and runner.
Sweeps prime (difficulty), num_operands, and collapse severity.
"""
import json
import argparse
import itertools
from pathlib import Path

try:
    from train import TrainConfig, train
except ImportError:
    from src.train import TrainConfig, train


def generate_manifest(output_dir: Path):
    """Generate the full Cartesian product into a manifest file."""
    manifest_path = output_dir / "manifest.json"

    # Grid 1: Difficulty Sweep
    primes = [13, 23, 31]
    operands = [2, 3]
    collapse_levels = [0.0, 0.05, 0.15, 0.30]
    seeds = [42, 43]

    tasks = []

    for p, op, lvl, seed in itertools.product(primes, operands, collapse_levels, seeds):
        tasks.append({
            "type": "difficulty",
            "prime": p,
            "num_operands": op,
            "collapse_level": lvl,
            "seed": seed,
            "d_model": 128,
            "n_heads": 4
        })

    # Grid 2: Capacity Control
    d_models = [64, 128, 256]
    for d, lvl, seed in itertools.product(d_models, collapse_levels, seeds):
        tasks.append({
            "type": "capacity",
            "prime": 31,
            "num_operands": 2,
            "collapse_level": lvl,
            "seed": seed,
            "d_model": d,
            "n_heads": max(1, d // 32)
        })

    # Remove exact duplicates just in case
    unique_tasks = []
    seen = set()
    for t in tasks:
        t_tup = tuple(sorted(t.items()))
        if t_tup not in seen:
            seen.add(t_tup)
            unique_tasks.append(t)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(unique_tasks, f, indent=2)
    return manifest_path, unique_tasks


def run_task(task: dict, output_dir: Path, max_steps: int):
    """Run a single task from the manifest."""
    condition_name = (
        f"{task['type']}_p{task['prime']}_op{task['num_operands']}"
        f"_lvl{task['collapse_level']}_d{task['d_model']}_seed{task['seed']}"
    )
    task_dir = output_dir / condition_name

    if (task_dir / "results.json").exists():
        print(f"Skipping {condition_name}, already exists.")
        return

    cfg = TrainConfig(
        prime=task['prime'],
        num_operands=task['num_operands'],
        d_model=task['d_model'],
        n_heads=task['n_heads'],
        collapse_level=task['collapse_level'],
        seed=task['seed'],
        max_steps=max_steps,
        condition_name=condition_name,
        output_dir=str(output_dir)
    )
    print(f"\nRunning {condition_name} ...")
    train(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/difficulty")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--array-id", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    manifest_path, tasks = generate_manifest(out_dir)
    print(f"Manifest generated at {manifest_path} with {len(tasks)} tasks.")

    if args.array_id is not None:
        if 0 <= args.array_id < len(tasks):
            run_task(tasks[args.array_id], out_dir, args.max_steps)
        else:
            print(f"array-id {args.array_id} out of bounds")
    else:
        for t in tasks:
            run_task(t, out_dir, args.max_steps)


if __name__ == "__main__":
    main()
